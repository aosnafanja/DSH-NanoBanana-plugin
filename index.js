/**
 * Nano Banana image generation for DeepSeek Harness.
 *
 * Registers two model-facing tools:
 * - `nano_banana` — generate or edit an image through the Gemini image models,
 *   saving the raster in the Session workspace and returning it inline.
 * - `nano_banana_models` — list the image models the configured key can reach.
 *
 * Deliberately free of Harness package imports so a locally installed bundle
 * resolves nothing outside Node's builtins.
 * @module @local/nano-banana
 */

import { mkdir, readFile, stat as statFile, writeFile } from 'node:fs/promises'
import { basename, dirname, extname, isAbsolute, join, relative, resolve as resolvePath } from 'node:path'
import {
  ASPECT_RATIOS,
  MODEL_ALIASES,
  buildImageRequest,
  generateImage,
  imageDimensions,
  listImageModels,
  modelLabel,
  resolveAspectRatio,
  resolveImageSize,
  resolveModel,
  resolveThinkingLevel,
} from './gemini.js'

export const name = 'nano-banana'
export const inject = ['tools']

/** Per-reference-image byte cap, and the cap on all reference bytes in one call. */
const MAX_REFERENCE_BYTES = 10 * 1024 * 1024
// Gemini caps the whole inline request near 20 MB, and inline data travels as
// base64 (+33%), so the raw budget stays below that after encoding.
const MAX_REFERENCES_BYTES = 13 * 1024 * 1024
/** Upper bound on reference images per call; the API rejects model-specific excess. */
const MAX_REFERENCES = 14

/** Latin stand-ins for Cyrillic letters, so Russian prompts keep a readable file name. */
const CYRILLIC = {
  'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e', 'ж': 'zh',
  'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
  'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts',
  'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu',
  'я': 'ya',
}

/** Image media type per file extension. */
const MEDIA_TYPES = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
}

/** Extension per generated media type. */
const EXTENSIONS = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp' }

/** Configuration defaults, overridden by the plugin row's `config`. */
const DEFAULTS = {
  apiKey: '',
  apiKeyEnv: 'GEMINI_API_KEY',
  baseUrl: 'https://generativelanguage.googleapis.com',
  defaultModel: 'flash',
  defaultSize: '2K',
  defaultAspectRatio: '',
  defaultThinking: '',
  outputDir: 'outputs/nano-banana',
  requestTimeoutMs: 240000,
}

/** Message shown when no key is configured anywhere. */
const MISSING_KEY_MESSAGE = 'no Gemini API key is configured for Nano Banana. '
  + 'Add "GEMINI_API_KEY: <key>" under refs: in ~/.dsh/.credentials.yaml (mode 600), '
  + 'then call the tool again. Create a key at https://aistudio.google.com/apikey'

/**
 * Merge row config over defaults.
 * @param {unknown} config - the row's `config` from the bundle patch.
 * @returns {typeof DEFAULTS} effective configuration.
 */
function effectiveConfig(config) {
  const source = config !== null && typeof config === 'object' ? config : {}
  const merged = { ...DEFAULTS }
  for (const [key, fallback] of Object.entries(DEFAULTS)) {
    const value = source[key]
    if (value === undefined || value === null) continue
    merged[key] = typeof fallback === 'number' ? Number(value) : String(value)
  }
  if (!Number.isFinite(merged.requestTimeoutMs) || merged.requestTimeoutMs <= 0) {
    merged.requestTimeoutMs = DEFAULTS.requestTimeoutMs
  }
  merged.baseUrl = merged.baseUrl.replace(/\/+$/u, '')
  return merged
}

/**
 * Resolve the Gemini API key: explicit config first, then the credential store,
 * then the launching environment.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context.
 * @param {typeof DEFAULTS} config - effective configuration.
 * @returns {Promise<string | undefined>} the key, when one is configured.
 */
async function resolveApiKey(ctx, config) {
  if (config.apiKey.trim() !== '') return config.apiKey.trim()
  const keyName = config.apiKeyEnv.trim() === '' ? DEFAULTS.apiKeyEnv : config.apiKeyEnv.trim()
  const credentials = ctx.get('credentials')
  if (credentials !== undefined) {
    try {
      const hit = await credentials.resolve(keyName)
      const value = typeof hit === 'string' ? hit : hit?.value
      if (typeof value === 'string' && value.trim() !== '') return value.trim()
    } catch {
      // A key name this store does not know falls through to the environment.
    }
  }
  const fromEnv = process.env[keyName]
  return typeof fromEnv === 'string' && fromEnv.trim() !== '' ? fromEnv.trim() : undefined
}

/**
 * Read the Session's current working directory.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context.
 * @param {{agent?: unknown, signal: AbortSignal}} exec - tool execution.
 * @returns {Promise<string | undefined>} absolute directory, when known.
 */
async function sessionCwd(ctx, exec) {
  const workingDirectory = ctx.get('workingDirectory')
  if (workingDirectory === undefined || exec.agent === undefined || exec.agent === null) return undefined
  try {
    return await workingDirectory.ensure(exec.agent, exec.signal)
  } catch {
    return undefined
  }
}

/**
 * Whether `candidate` stays inside `root`.
 * @param {string} root - absolute directory.
 * @param {string} candidate - absolute path.
 * @returns {boolean} true when contained.
 */
function isInside(root, candidate) {
  const rel = relative(resolvePath(root), resolvePath(candidate))
  return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel))
}

/**
 * Resolve an existing regular file through the filesystem service, falling back
 * to plain Node resolution when no backend is mounted.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context.
 * @param {{agent?: unknown, signal: AbortSignal}} exec - tool execution.
 * @param {string} requested - model-supplied path.
 * @returns {Promise<{abs: string, size: number, target?: unknown}>} resolved file facts.
 */
async function resolveExistingFile(ctx, exec, requested) {
  const fsService = ctx.get('fs')
  if (fsService !== undefined) {
    const cwd = await sessionCwd(ctx, exec)
    const target = await fsService.resolve(requested, { ...cwd === undefined ? {} : { cwd }, signal: exec.signal })
    const info = await fsService.stat(target, exec.signal)
    if (info === undefined) throw new Error(`cannot read "${requested}": file not found`)
    if (info.type !== 'file') throw new Error(`cannot read "${requested}": not a regular file`)
    return { abs: fsService.processPath(target), size: typeof info.size === 'number' ? info.size : 0, target }
  }
  const abs = resolvePath(requested)
  let info
  try {
    info = await statFile(abs)
  } catch {
    throw new Error(`cannot read "${requested}": file not found`)
  }
  if (!info.isFile()) throw new Error(`cannot read "${requested}": not a regular file`)
  return { abs, size: info.size }
}

/**
 * Read raw bytes of a resolved file.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context.
 * @param {{abs: string, target?: unknown}} file - resolved file.
 * @param {AbortSignal} signal - cancellation.
 * @param {number} maxBytes - inclusive byte cap.
 * @returns {Promise<Uint8Array>} file bytes.
 */
async function readFileBytes(ctx, file, signal, maxBytes) {
  const fsService = ctx.get('fs')
  if (fsService !== undefined && file.target !== undefined) {
    return fsService.readBytes(file.target, signal, maxBytes)
  }
  return readFile(file.abs, { signal })
}

/**
 * Identify a supported image media type from extension, then signature.
 * @param {string} requestedPath - model-supplied path.
 * @param {Uint8Array} bytes - file bytes.
 * @returns {string | undefined} supported media type, when the bytes are an image.
 */
function mediaTypeOf(requestedPath, bytes) {
  const declared = MEDIA_TYPES[extname(requestedPath).toLowerCase()]
  if (declared !== undefined) return declared
  if (bytes.byteLength >= 8
    && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) return 'image/png'
  if (bytes.byteLength >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'image/jpeg'
  if (bytes.byteLength >= 12
    && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46
    && bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50) return 'image/webp'
  return undefined
}

/**
 * Load reference images for one request.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context.
 * @param {{agent?: unknown, signal: AbortSignal}} exec - tool execution.
 * @param {unknown} references - `references` argument.
 * @returns {Promise<{images: {data: string, mimeType: string}[], paths: string[]}>} inline images and display paths.
 */
async function loadReferences(ctx, exec, references) {
  if (references !== undefined && references !== null && !Array.isArray(references)) {
    throw new Error('references must be an array of file paths')
  }
  const requested = Array.isArray(references)
    ? references.filter(value => typeof value === 'string' && value.trim() !== '')
    : []
  if (requested.length === 0) return { images: [], paths: [] }
  if (requested.length > MAX_REFERENCES) {
    throw new Error(`at most ${MAX_REFERENCES} reference images are supported, got ${requested.length}`)
  }
  const images = []
  const paths = []
  let total = 0
  for (const reference of requested) {
    const file = await resolveExistingFile(ctx, exec, reference)
    if (file.size > MAX_REFERENCE_BYTES) {
      throw new Error(`reference image "${reference}" is ${file.size} bytes; the per-image limit is ${MAX_REFERENCE_BYTES}`)
    }
    const bytes = await readFileBytes(ctx, file, exec.signal, MAX_REFERENCE_BYTES)
    total += bytes.byteLength
    if (total > MAX_REFERENCES_BYTES) {
      throw new Error(`reference images exceed the ${MAX_REFERENCES_BYTES}-byte request budget; send fewer or smaller images`)
    }
    const mimeType = mediaTypeOf(reference, bytes)
    if (mimeType === undefined) {
      throw new Error(`reference image "${reference}" is not a PNG, JPEG, or WebP image`)
    }
    images.push({ data: Buffer.from(bytes).toString('base64'), mimeType })
    paths.push(file.abs)
  }
  return { images, paths }
}

/**
 * Turn a prompt into a short ASCII file-name fragment.
 * @param {{name?: unknown, prompt: string}} options - naming inputs.
 * @returns {string} slug safe for a file name.
 */
function slugFor({ name, prompt }) {
  const source = typeof name === 'string' && name.trim() !== '' ? name : prompt
  const transliterated = source
    .toLowerCase()
    .replace(/[а-яё]/gu, letter => CYRILLIC[letter] ?? '')
  const slug = transliterated
    .replace(/[^a-z0-9]+/gu, '-')
    .replace(/^-+|-+$/gu, '')
    .slice(0, 48)
    .replace(/-+$/u, '')
  return slug === '' ? 'nano-banana' : slug
}

/**
 * Local timestamp used in generated file names.
 * @returns {string} `YYMMDD-HHmmss`.
 */
function timestamp() {
  const now = new Date()
  const pad = value => String(value).padStart(2, '0')
  return `${String(now.getFullYear()).slice(2)}${pad(now.getMonth() + 1)}${pad(now.getDate())}`
    + `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
}

/**
 * Resolve the destination of a generated image and write its bytes.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context.
 * @param {{agent?: unknown, signal: AbortSignal}} exec - tool execution.
 * @param {object} options - write options.
 * @param {string} options.targetPath - requested or configured destination.
 * @param {string | undefined} options.cwd - Session working directory, when known.
 * @param {boolean} options.modelChosen - whether the model supplied the path.
 * @param {Uint8Array} options.bytes - image bytes.
 * @returns {Promise<{path: string, relativePath: string}>} written locations.
 */
async function writeOutput(ctx, exec, { targetPath, cwd, modelChosen, bytes }) {
  const fsService = ctx.get('fs')
  const root = cwd ?? process.cwd()
  let abs
  if (fsService !== undefined) {
    try {
      const resolved = await fsService.resolve(targetPath, {
        ...cwd === undefined ? {} : { cwd },
        signal: exec.signal,
      })
      abs = fsService.processPath(resolved)
    } catch {
      // A backend that will not resolve a not-yet-created target must not cost
      // the caller an image that was already generated and paid for.
      abs = isAbsolute(targetPath) ? targetPath : resolvePath(root, targetPath)
    }
  } else {
    abs = isAbsolute(targetPath) ? targetPath : resolvePath(root, targetPath)
  }
  if (modelChosen && cwd !== undefined && !isInside(cwd, abs)) {
    throw new Error(`refusing to write "${abs}": the output must stay inside the Session workspace (${cwd})`)
  }
  await mkdir(dirname(abs), { recursive: true })
  abs = await writeUnique(abs, bytes)
  return {
    path: abs,
    relativePath: cwd !== undefined && isInside(cwd, abs) ? relative(cwd, abs) : abs,
  }
}

/**
 * Write bytes to the first free file name, never clobbering an existing image.
 * The `wx` flag makes the existence check part of the write, so two parallel
 * generations cannot both pick the same path.
 * @param {string} abs - desired absolute path.
 * @param {Uint8Array} bytes - image bytes.
 * @returns {Promise<string>} the path actually written.
 */
async function writeUnique(abs, bytes) {
  const extension = extname(abs)
  const stem = extension === '' ? abs : abs.slice(0, abs.length - extension.length)
  for (let index = 1; index <= 99; index += 1) {
    const candidate = index === 1 ? abs : `${stem}-${index}${extension}`
    try {
      await writeFile(candidate, bytes, { flag: 'wx' })
      return candidate
    } catch (error) {
      if (error?.code === 'EEXIST') continue
      throw error
    }
  }
  throw new Error(`cannot find a free file name beside "${abs}"; remove or rename the existing files first`)
}

/**
 * Combine the caller's cancellation with a request deadline.
 * @param {AbortSignal} signal - caller cancellation.
 * @param {number} timeoutMs - deadline.
 * @returns {{signal: AbortSignal, dispose: () => void, timedOut: () => boolean}} combined signal.
 */
function withDeadline(signal, timeoutMs) {
  const controller = new AbortController()
  let timedOut = false
  const onAbort = () => controller.abort(signal.reason)
  if (signal.aborted) controller.abort(signal.reason)
  else signal.addEventListener('abort', onAbort, { once: true })
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort(new DOMException('request deadline exceeded', 'TimeoutError'))
  }, timeoutMs)
  if (typeof timer.unref === 'function') timer.unref()
  return {
    signal: controller.signal,
    timedOut: () => timedOut,
    dispose: () => {
      clearTimeout(timer)
      signal.removeEventListener('abort', onAbort)
    },
  }
}

/**
 * Human byte size.
 * @param {number} bytes - byte count.
 * @returns {string} `12.3 MB` or `480 KB`.
 */
function formatBytes(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${Math.max(1, Math.round(bytes / 1024))} KB`
}

/** Attachment object schema stored beside the image path. */
const IMAGE_VALUE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['attachmentId', 'mediaType', 'bytes', 'width', 'height'],
  properties: {
    attachmentId: { type: 'string' },
    mediaType: { type: 'string', enum: ['image/png', 'image/jpeg', 'image/webp'] },
    bytes: { type: 'integer' },
    width: { type: 'integer' },
    height: { type: 'integer' },
    name: { type: 'string' },
  },
}

/**
 * Render one completed generation as model-facing content plus its image.
 * @param {object} value - canonical tool value.
 * @returns {object[]} content blocks.
 */
function renderGeneration(value) {
  const label = modelLabel(value.model)
  const shortLabel = label.includes(' (') ? label.slice(0, label.indexOf(' (')) : label
  const facts = [
    shortLabel === value.model ? value.model : `${value.model} · ${shortLabel}`,
    value.width === undefined ? undefined : `${value.width}x${value.height} px`,
    formatBytes(value.bytes),
    value.aspectRatio,
    value.size,
    `${(value.durationMs / 1000).toFixed(1)} s`,
    value.image === undefined || value.width === undefined
      || (value.image.width === value.width && value.image.height === value.height)
      ? undefined
      : `preview downscaled to ${value.image.width}x${value.image.height}`,
  ].filter(part => part !== undefined && part !== '').join(' · ')
  const lines = [
    `<path>${value.path}</path>`,
    value.relativePath === value.path ? undefined : `<relative>${value.relativePath}</relative>`,
    `<type>image</type>`,
    '<content>',
    facts,
    value.references.length === 0 ? undefined : `references: ${value.references.join(', ')}`,
    value.text,
    '</content>',
  ].filter(line => line !== undefined)
  const content = [{ type: 'text', text: lines.join('\n') }]
  if (value.image !== undefined) {
    content.push({
      type: 'image',
      attachment: {
        attachmentId: value.image.attachmentId,
        mediaType: value.image.mediaType,
        bytes: value.image.bytes,
        width: value.image.width,
        height: value.image.height,
        ...value.image.name === undefined ? {} : { name: value.image.name },
      },
    })
  }
  return content
}

/**
 * Build the generation tool for one plugin instance.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context.
 * @param {typeof DEFAULTS} config - effective configuration.
 * @returns {object} tool definition.
 */
function createGenerationTool(ctx, config) {
  return {
    name: 'nano_banana',
    description: 'Generate or edit an image with Google Nano Banana (Gemini image models). '
      + 'Describe the scene in a prompt; pass reference images to edit, compose, or restyle them. '
      + 'The image is written into the workspace and returned in this result, so no follow-up read is needed. '
      + 'One image per call; call again for more variants.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      required: ['prompt'],
      properties: {
        prompt: {
          type: 'string',
          description: 'What to draw, or how to change the reference images. Describe the scene rather than listing keywords.',
        },
        references: {
          type: 'array',
          items: { type: 'string' },
          description: 'Workspace-relative or absolute paths of PNG/JPEG/WebP images to edit, compose, or restyle (up to 14).',
        },
        model: {
          type: 'string',
          description: `Alias (flash, 2.1, lite, pro, legacy) or a full Gemini image model id. Default: ${config.defaultModel}.`,
        },
        size: {
          type: 'string',
          enum: ['512', '1K', '2K', '4K'],
          description: `Output resolution. Default ${config.defaultSize}. 512 works only on gemini-3.1-flash-image; the legacy model is fixed at 1K.`,
        },
        aspect_ratio: {
          type: 'string',
          enum: ASPECT_RATIOS,
          description: 'Output aspect ratio. Default: the model decides from the prompt.',
        },
        thinking: {
          type: 'string',
          enum: ['minimal', 'medium', 'high'],
          description: 'Reasoning effort for models that support it; ignored elsewhere. Lower is faster and cheaper.',
        },
        name: {
          type: 'string',
          description: 'File-name stem for the saved image. Default: derived from the prompt.',
        },
        path: {
          type: 'string',
          description: 'Workspace-relative output path including the file name; overrides name and the output directory. Default: the output directory plus name and timestamp.',
        },
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['path', 'relativePath', 'model', 'bytes', 'durationMs', 'references'],
        properties: {
          path: { type: 'string', description: 'Absolute path in the filesystem execution world.' },
          relativePath: { type: 'string', description: 'Path relative to the Session working directory.' },
          model: { type: 'string' },
          aspectRatio: { type: 'string' },
          size: { type: 'string' },
          width: { type: 'integer' },
          height: { type: 'integer' },
          bytes: { type: 'integer' },
          durationMs: { type: 'integer' },
          text: { type: 'string' },
          references: { type: 'array', items: { type: 'string' } },
          image: IMAGE_VALUE_SCHEMA,
        },
      },
      render: (_args, value) => renderGeneration(value),
      presentationMeta: (_args, value) => ({ path: value.path, relativePath: value.relativePath }),
    },
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const started = Date.now()
      const apiKey = await resolveApiKey(ctx, config)
      if (apiKey === undefined) throw new Error(MISSING_KEY_MESSAGE)

      const prompt = typeof args.prompt === 'string' ? args.prompt.trim() : ''
      if (prompt === '') throw new Error('prompt must be a non-empty string')

      let defaultModel = MODEL_ALIASES.flash
      try {
        defaultModel = resolveModel(config.defaultModel, MODEL_ALIASES.flash)
      } catch {
        defaultModel = MODEL_ALIASES.flash
      }
      const model = resolveModel(args.model, defaultModel)
      const imageSize = resolveImageSize(model, args.size ?? config.defaultSize)
      const aspectRatio = resolveAspectRatio(args.aspect_ratio ?? config.defaultAspectRatio)
      const thinkingLevel = resolveThinkingLevel(model, args.thinking ?? config.defaultThinking)
      const references = await loadReferences(ctx, exec, args.references)
      const body = buildImageRequest({
        prompt,
        images: references.images,
        model,
        aspectRatio,
        imageSize,
        thinkingLevel,
      })

      const deadline = withDeadline(exec.signal, config.requestTimeoutMs)
      let generated
      try {
        generated = await generateImage({ apiKey, baseUrl: config.baseUrl, model, body, signal: deadline.signal })
      } catch (error) {
        if (exec.signal.aborted) throw error
        if (deadline.timedOut()) {
          throw new Error(`image generation exceeded ${config.requestTimeoutMs} ms; raise requestTimeoutMs in the plugin config or use a faster model`)
        }
        throw error
      } finally {
        deadline.dispose()
      }

      const extension = EXTENSIONS[generated.mimeType] ?? 'png'
      const cwd = await sessionCwd(ctx, exec)
      const outputDir = config.outputDir.trim() === '' ? DEFAULTS.outputDir : config.outputDir.trim()
      const modelPath = typeof args.path === 'string' && args.path.trim() !== '' ? args.path.trim() : undefined
      const targetPath = modelPath ?? join(outputDir, `${slugFor({ name: args.name, prompt })}-${timestamp()}.${extension}`)
      const written = await writeOutput(ctx, exec, {
        targetPath,
        cwd,
        modelChosen: modelPath !== undefined,
        bytes: generated.bytes,
      })
      const dimensions = imageDimensions(generated.bytes)

      let image
      const attachments = ctx.get('attachments')
      if (attachments !== undefined) {
        try {
          const ref = await attachments.saveImage({
            data: generated.bytes,
            mediaType: generated.mimeType,
            name: basename(written.path),
          })
          image = {
            attachmentId: ref.attachmentId,
            mediaType: ref.mediaType,
            bytes: ref.bytes,
            width: ref.width,
            height: ref.height,
            ...ref.name === undefined ? {} : { name: ref.name },
          }
        } catch {
          // A preview the attachment store refuses (for example a 4K image over
          // its byte budget) must not fail a generation already written to disk.
        }
      }

      return {
        path: written.path,
        relativePath: written.relativePath,
        model,
        ...aspectRatio === undefined ? {} : { aspectRatio },
        ...imageSize === undefined ? {} : { size: imageSize },
        ...dimensions === undefined ? {} : { width: dimensions.width, height: dimensions.height },
        bytes: generated.bytes.byteLength,
        durationMs: Date.now() - started,
        ...generated.text === undefined ? {} : { text: generated.text },
        references: references.paths,
        ...image === undefined ? {} : { image },
      }
    },
  }
}

/**
 * Build the model-listing tool for one plugin instance.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context.
 * @param {typeof DEFAULTS} config - effective configuration.
 * @returns {object} tool definition.
 */
function createModelsTool(ctx, config) {
  return {
    name: 'nano_banana_models',
    description: 'List the Google image-generation models the Gemini API advertises for the configured key, '
      + 'with the resolutions each accepts. The list is advisory: some image models work when called by id '
      + 'even though the API does not advertise them.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {},
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['count', 'models'],
        properties: {
          count: { type: 'integer' },
          defaultModel: { type: 'string' },
          models: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['id', 'label'],
              properties: {
                id: { type: 'string' },
                label: { type: 'string' },
                description: { type: 'string' },
                listed: { type: 'boolean' },
                sizes: { type: 'array', items: { type: 'string' } },
              },
            },
          },
        },
      },
      render: (_args, value) => {
        if (value.count === 0) {
          return [{
            type: 'text',
            text: 'The API key reaches no image models. Check that the key is valid and that the Generative Language API is enabled.',
          }]
        }
        const lines = value.models.map(model => `- ${model.id} — ${model.label}`
          + `${model.sizes === undefined ? '' : ` (sizes: ${model.sizes.join(', ')})`}`
          + `${model.listed === false ? ' (not advertised, callable by id)' : ''}`)
        const header = value.defaultModel === undefined
          ? `${value.count} image models:`
          : `${value.count} image models (configured default: ${value.defaultModel}):`
        return [{ type: 'text', text: `${header}\n${lines.join('\n')}` }]
      },
    },
    isConcurrencySafe: () => true,
    async execute(_args, exec) {
      const apiKey = await resolveApiKey(ctx, config)
      if (apiKey === undefined) throw new Error(MISSING_KEY_MESSAGE)
      const deadline = withDeadline(exec.signal, Math.min(config.requestTimeoutMs, 60000))
      try {
        const models = await listImageModels({ apiKey, baseUrl: config.baseUrl, signal: deadline.signal })
        return { count: models.length, models, defaultModel: config.defaultModel }
      } catch (error) {
        if (exec.signal.aborted) throw error
        if (deadline.timedOut()) throw new Error('listing image models timed out')
        throw error
      } finally {
        deadline.dispose()
      }
    },
  }
}

/**
 * Register the Nano Banana tools.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context.
 * @param {unknown} config - row config from the bundle patch.
 * @returns {() => void} disposer for both registrations.
 */
export function apply(ctx, config) {
  const effective = effectiveConfig(config)
  const disposers = [
    ctx.tools.register(createGenerationTool(ctx, effective)),
    ctx.tools.register(createModelsTool(ctx, effective)),
  ]
  return () => {
    for (const dispose of disposers) {
      try {
        dispose()
      } catch {
        // Already disposed with the plugin fiber.
      }
    }
  }
}
