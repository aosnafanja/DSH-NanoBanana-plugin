/**
 * Minimal Gemini (Nano Banana) image client: model aliases, request shaping,
 * response unwrapping, and error translation. No Harness imports, so this
 * module stays a plain Node/Electron-friendly module.
 * @module @local/nano-banana/gemini
 */

/** Alias -> canonical Gemini model id. */
export const MODEL_ALIASES = {
  'flash': 'gemini-3.1-flash-image',
  'nano-banana-2': 'gemini-3.1-flash-image',
  '2': 'gemini-3.1-flash-image',
  '2.1': 'gemini-nano-banana-2.1',
  'nano-banana-2.1': 'gemini-nano-banana-2.1',
  'banana': 'gemini-nano-banana-2.1',
  'workhorse': 'gemini-nano-banana-2.1',
  'lite': 'gemini-3.1-flash-lite-image',
  'nano-banana-2-lite': 'gemini-3.1-flash-lite-image',
  'pro': 'gemini-3-pro-image',
  'nano-banana-pro': 'gemini-3-pro-image',
  'legacy': 'gemini-2.5-flash-image',
  'nano-banana': 'gemini-2.5-flash-image',
}

/** Human-facing label per canonical id. */
export const MODEL_LABELS = {
  'gemini-3.1-flash-image': 'Nano Banana 2 (Gemini 3.1 Flash Image)',
  'gemini-nano-banana-2.1': 'Nano Banana 2.1 (Gemini Nano Banana 2.1)',
  'gemini-3.1-flash-lite-image': 'Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)',
  'gemini-3-pro-image': 'Nano Banana Pro (Gemini 3 Pro Image)',
  'gemini-2.5-flash-image': 'Nano Banana (Gemini 2.5 Flash Image, legacy)',
}

/** Resolutions each known model accepts. Unknown ids pass through unchecked. */
export const MODEL_SIZES = {
  'gemini-3.1-flash-image': ['512', '1K', '2K', '4K'],
  'gemini-nano-banana-2.1': ['1K', '2K', '4K'],
  'gemini-3.1-flash-lite-image': ['1K', '2K', '4K'],
  'gemini-3-pro-image': ['1K', '2K', '4K'],
  'gemini-2.5-flash-image': ['1K'],
}

/** Aspect ratios the image models accept. */
export const ASPECT_RATIOS = [
  '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9',
  '1:4', '4:1', '1:8', '8:1',
]

/** Thinking levels each model accepts; models missing here take no level. */
const THINKING_LEVELS = {
  'gemini-nano-banana-2.1': ['minimal', 'medium', 'high'],
  'gemini-3.1-flash-image': ['minimal', 'high'],
  'gemini-3.1-flash-lite-image': ['minimal', 'high'],
}

/**
 * Resolve a model alias or raw id to a canonical Gemini model id.
 * @param {unknown} value - alias such as `flash`, or a full `gemini-*` id.
 * @param {string} fallback - id used when `value` is absent.
 * @returns {string} canonical model id.
 */
export function resolveModel(value, fallback) {
  const raw = typeof value === 'string' ? value.trim() : ''
  if (raw === '') return fallback
  const alias = MODEL_ALIASES[raw.toLowerCase()]
  if (alias !== undefined) return alias
  if (/^[a-z0-9][a-z0-9.\-]*$/i.test(raw) && raw.includes('gemini')) return raw
  throw new Error(
    `unknown model "${raw}"; use an alias (${Object.keys(MODEL_ALIASES).slice(0, 6).join(', ')}) or a full Gemini model id`,
  )
}

/**
 * Describe the human-facing label for a model id.
 * @param {string} id - canonical model id.
 * @returns {string} label, or the id itself for unknown models.
 */
export function modelLabel(id) {
  return MODEL_LABELS[id] ?? id
}

/**
 * Normalize a requested resolution for one model.
 * @param {string} model - canonical model id.
 * @param {unknown} requested - `512`, `1K`, `2K`, or `4K`.
 * @returns {string | undefined} the value to send, or undefined to omit it.
 * @throws {Error} when the model does not support the requested resolution.
 */
export function resolveImageSize(model, requested) {
  const supported = MODEL_SIZES[model]
  const asked = requested === undefined || requested === null || requested === ''
    ? supported?.[0] ?? '1K'
    : String(requested).trim().toUpperCase()
  if (supported !== undefined && !supported.includes(asked)) {
    throw new Error(
      `model "${model}" does not support size "${requested}"; supported sizes: ${supported.join(', ')}`,
    )
  }
  // The legacy 2.5 model has one fixed resolution and rejects an explicit
  // `imageSize`, so the setting is omitted rather than sent.
  return model === 'gemini-2.5-flash-image' ? undefined : asked
}

/**
 * Validate a requested aspect ratio.
 * @param {unknown} requested - ratio such as `4:5`.
 * @returns {string | undefined} the ratio to send, or undefined to omit it.
 * @throws {Error} when the ratio is not one the API accepts.
 */
export function resolveAspectRatio(requested) {
  if (requested === undefined || requested === null || requested === '') return undefined
  const value = String(requested).trim()
  if (!ASPECT_RATIOS.includes(value)) {
    throw new Error(`aspect_ratio "${requested}" is not supported; use one of ${ASPECT_RATIOS.join(', ')}`)
  }
  return value
}

/**
 * Normalize a thinking level for the models that expose the setting.
 * @param {string} model - canonical model id.
 * @param {unknown} requested - `minimal`, `medium`, or `high`.
 * @returns {string | undefined} the uppercase enum value, or undefined to omit it.
 */
export function resolveThinkingLevel(model, requested) {
  if (requested === undefined || requested === null || requested === '') return undefined
  const levels = THINKING_LEVELS[model]
  if (levels === undefined) return undefined
  const value = String(requested).trim().toLowerCase()
  if (!levels.includes(value)) {
    throw new Error(`model "${model}" does not support thinking "${requested}"; use ${levels.join(', ')}`)
  }
  return value.toUpperCase()
}

/**
 * Build the `generateContent` request body for an image turn.
 * @param {object} options - request options.
 * @param {string} options.prompt - text instruction.
 * @param {{data: string, mimeType: string}[]} options.images - inline reference images.
 * @param {string} options.model - canonical model id.
 * @param {string} [options.aspectRatio] - validated ratio.
 * @param {string} [options.imageSize] - validated resolution.
 * @param {string} [options.thinkingLevel] - uppercase thinking enum value.
 * @returns {object} JSON request body.
 */
export function buildImageRequest({ prompt, images, model, aspectRatio, imageSize, thinkingLevel }) {
  const parts = [{ text: prompt }]
  for (const image of images) {
    parts.push({ inline_data: { mime_type: image.mimeType, data: image.data } })
  }
  const generationConfig = { responseModalities: ['TEXT', 'IMAGE'] }
  if (aspectRatio !== undefined || imageSize !== undefined) {
    generationConfig.imageConfig = {
      ...aspectRatio === undefined ? {} : { aspectRatio },
      ...imageSize === undefined ? {} : { imageSize },
    }
  }
  if (thinkingLevel !== undefined) {
    generationConfig.thinkingConfig = { thinkingLevel }
  }
  return { contents: [{ role: 'user', parts }], generationConfig }
}

/**
 * Describe one HTTP failure with an actionable hint.
 * @param {number} status - HTTP status.
 * @param {{error?: {message?: string, status?: string, details?: unknown[]}}} body - parsed error body.
 * @returns {string} human-facing message.
 */
export function describeApiError(status, body) {
  const message = body?.error?.message ?? `HTTP ${status}`
  const statusText = body?.error?.status ?? ''
  const compact = message.length > 600 ? `${message.slice(0, 600)}…` : message
  if (status === 400 && /API key not valid/i.test(message)) {
    return `${compact} — the Gemini API key is invalid; set the correct GEMINI_API_KEY in the plugin config`
  }
  if (status === 403) return `${compact} — the key is not allowed to use this model or API (enable the Generative Language API / check key restrictions)`
  if (status === 404) return `${compact} — the model may not be available for this key; run nano_banana_models to list what is available`
  if (status === 429) return `${compact} — rate limit or quota exhausted for this key`
  if (status >= 500) return `${compact} — Gemini API side failure (${status} ${statusText}); retry shortly`
  return `${compact}${statusText === '' ? '' : ` (${statusText})`}`
}

/**
 * Call `generateContent` and unwrap the first usable image.
 * @param {object} options - call options.
 * @param {string} options.apiKey - resolved API key.
 * @param {string} options.baseUrl - API root without a trailing slash.
 * @param {string} options.model - canonical model id.
 * @param {object} options.body - request body from {@link buildImageRequest}.
 * @param {AbortSignal} options.signal - cancellation for the HTTP call.
 * @returns {Promise<{bytes: Uint8Array, mimeType: string, text?: string, finishReason?: string}>} generated image.
 * @throws {Error} on HTTP failure or a response without an image.
 */
export async function generateImage({ apiKey, baseUrl, model, body, signal }) {
  const url = `${baseUrl}/v1/models/${encodeURIComponent(model)}:generateContent`
  let response
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'x-goog-api-key': apiKey, 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
  } catch (error) {
    if (error?.name === 'AbortError' || error?.name === 'TimeoutError') throw error
    throw new Error(`cannot reach the Gemini API at ${baseUrl}: ${error?.message ?? String(error)}`)
  }
  const raw = await response.text()
  let payload
  try {
    payload = raw === '' ? {} : JSON.parse(raw)
  } catch {
    payload = { error: { message: `unexpected non-JSON response (${raw.slice(0, 200)})` } }
  }
  if (!response.ok) throw new Error(describeApiError(response.status, payload))

  const candidate = payload?.candidates?.[0]
  const blockReason = payload?.promptFeedback?.blockReason
  if (blockReason !== undefined && blockReason !== null) {
    throw new Error(`the request was blocked by the API (${blockReason}); rephrase the prompt`)
  }
  const parts = candidate?.content?.parts ?? []
  let text
  let image
  for (const part of parts) {
    const inline = part.inlineData ?? part.inline_data
    if (part.thought === true) continue
    if (inline?.data) {
      image = { data: inline.data, mimeType: inline.mimeType ?? inline.mime_type ?? 'image/png' }
      continue
    }
    if (typeof part.text === 'string' && part.text.trim() !== '') {
      text = text === undefined ? part.text.trim() : `${text}\n${part.text.trim()}`
    }
  }
  if (image === undefined) {
    const reason = candidate?.finishReason
    const detail = reason === undefined ? 'no image in the response' : `no image in the response (finishReason: ${reason})`
    const hint = reason === 'SAFETY' || reason === 'PROHIBITED_CONTENT' || reason === 'IMAGE_SAFETY'
      ? '; the prompt or a reference image was refused by safety filters'
      : ''
    throw new Error(`${detail}${hint}${text === undefined ? '' : `; model said: ${text.slice(0, 400)}`}`)
  }
  return {
    bytes: Buffer.from(image.data, 'base64'),
    mimeType: image.mimeType,
    text,
    finishReason: candidate?.finishReason,
  }
}

/**
 * List the image-capable models the key can reach.
 * @param {object} options - call options.
 * @param {string} options.apiKey - resolved API key.
 * @param {string} options.baseUrl - API root without a trailing slash.
 * @param {AbortSignal} options.signal - cancellation for the HTTP call.
 * @returns {Promise<{id: string, label: string, description?: string, sizes?: string[]}[]>} image models.
 */
export async function listImageModels({ apiKey, baseUrl, signal }) {
  const models = []
  let pageToken
  for (let page = 0; page < 5; page += 1) {
    const url = new URL(`${baseUrl}/v1/models`)
    url.searchParams.set('pageSize', '200')
    if (pageToken !== undefined) url.searchParams.set('pageToken', pageToken)
    const response = await fetch(url, { headers: { 'x-goog-api-key': apiKey }, signal })
    const raw = await response.text()
    let payload
    try {
      payload = raw === '' ? {} : JSON.parse(raw)
    } catch {
      payload = { error: { message: `unexpected non-JSON response (${raw.slice(0, 200)})` } }
    }
    if (!response.ok) throw new Error(describeApiError(response.status, payload))
    for (const model of payload.models ?? []) {
      const id = String(model.name ?? '').replace(/^models\//, '')
      const methods = model.supportedGenerationMethods
      if (Array.isArray(methods) && !methods.includes('generateContent')) continue
      if (!/image/i.test(id) && !/image/i.test(String(model.displayName ?? ''))) continue
      models.push({
        id,
        label: MODEL_LABELS[id] ?? String(model.displayName ?? id),
        ...typeof model.description === 'string' && model.description !== '' ? { description: model.description } : {},
        ...MODEL_SIZES[id] === undefined ? {} : { sizes: MODEL_SIZES[id] },
      })
    }
    pageToken = payload.nextPageToken
    if (pageToken === undefined || pageToken === null || pageToken === '') break
  }
  // The API does not advertise every callable image model (for example
  // gemini-nano-banana-2.1 answers calls while missing from models.list), so the
  // catalogue of known image models fills the gaps instead of hiding them.
  for (const id of Object.keys(MODEL_LABELS)) {
    if (models.some(model => model.id === id)) continue
    models.push({
      id,
      label: MODEL_LABELS[id],
      listed: false,
      ...MODEL_SIZES[id] === undefined ? {} : { sizes: MODEL_SIZES[id] },
    })
  }
  models.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
  return models
}

/**
 * Read the pixel dimensions out of PNG bytes.
 * @param {Uint8Array} bytes - image bytes.
 * @returns {{width: number, height: number} | undefined} dimensions when the bytes are PNG.
 */
export function pngDimensions(bytes) {
  if (bytes.byteLength < 24) return undefined
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
  for (let index = 0; index < signature.length; index += 1) {
    if (bytes[index] !== signature[index]) return undefined
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  return { width: view.getUint32(16), height: view.getUint32(20) }
}

/**
 * Read the pixel dimensions out of JPEG bytes by scanning for a start-of-frame segment.
 * @param {Uint8Array} bytes - image bytes.
 * @returns {{width: number, height: number} | undefined} dimensions when the bytes are JPEG.
 */
export function jpegDimensions(bytes) {
  if (bytes.byteLength < 12 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return undefined
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let offset = 2
  while (offset + 9 < bytes.byteLength) {
    if (bytes[offset] !== 0xff) {
      offset += 1
      continue
    }
    const marker = bytes[offset + 1]
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      offset += 2
      continue
    }
    const length = view.getUint16(offset + 2)
    const isFrame = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc
    if (isFrame) return { height: view.getUint16(offset + 5), width: view.getUint16(offset + 7) }
    if (length < 2) return undefined
    offset += 2 + length
  }
  return undefined
}

/**
 * Read the pixel dimensions out of WebP bytes.
 * @param {Uint8Array} bytes - image bytes.
 * @returns {{width: number, height: number} | undefined} dimensions when the bytes are WebP.
 */
export function webpDimensions(bytes) {
  if (bytes.byteLength < 30) return undefined
  const ascii = (offset, value) => {
    for (let index = 0; index < value.length; index += 1) {
      if (bytes[offset + index] !== value.charCodeAt(index)) return false
    }
    return true
  }
  if (!ascii(0, 'RIFF') || !ascii(8, 'WEBP')) return undefined
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (ascii(12, 'VP8X')) {
    return {
      width: 1 + (bytes[24] | (bytes[25] << 8) | (bytes[26] << 16)),
      height: 1 + (bytes[27] | (bytes[28] << 8) | (bytes[29] << 16)),
    }
  }
  if (ascii(12, 'VP8L') && bytes[20] === 0x2f) {
    const bits = bytes[21] | (bytes[22] << 8) | (bytes[23] << 16) | (bytes[24] << 24)
    return { width: 1 + (bits & 0x3fff), height: 1 + ((bits >> 14) & 0x3fff) }
  }
  if (ascii(12, 'VP8 ') && bytes.byteLength >= 30) {
    return { width: view.getUint16(26, true) & 0x3fff, height: view.getUint16(28, true) & 0x3fff }
  }
  return undefined
}

/**
 * Read the pixel dimensions of any image the API returns.
 * @param {Uint8Array} bytes - image bytes.
 * @returns {{width: number, height: number} | undefined} dimensions, when the format is recognized.
 */
export function imageDimensions(bytes) {
  return pngDimensions(bytes) ?? jpegDimensions(bytes) ?? webpDimensions(bytes)
}
