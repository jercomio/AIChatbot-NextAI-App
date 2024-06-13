import { Tiktoken } from "@dqbd/tiktoken"
import cl100k_base from "@dqbd/tiktoken/encoders/cl100k_base.json"
import OpenAI from "openai"
import dotenv from "dotenv"
import fs from "fs/promises"
import { neon } from "@neondatabase/serverless"
import path from "path"
import { logger } from "./utils"
dotenv.config()

const databaseUrl = process.env.DATABASE_URL
const openaiApiKey = process.env.OPENAI_API_KEY

if (!databaseUrl || !openaiApiKey) {
  throw new Error('Missing environment variables')
}

const openai = new OpenAI({
    apiKey: openaiApiKey,
})
const sql = neon(databaseUrl)

const encoding = new Tiktoken(
    cl100k_base.bpe_ranks,
    cl100k_base.special_tokens,
    cl100k_base.pat_str
)

// Step 1: Process the files

type TextFile = {
    filePath: string
    text: string
}

async function processFiles(folder: string): Promise<TextFile[]> {
    const files: TextFile[] = []

    const folderPath = `./data/${folder}`

    const entries = await fs.readdir(folderPath, { withFileTypes: true })

    for (const entry of entries) {
        const fullPath = path.join(folderPath, entry.name)

        if (entry.isDirectory()) {
            continue
        }

        const text = await fs.readFile(fullPath, "utf-8")

        files.push({
            filePath: entry.name,
            text,
        })
    }
    return files
}

// Step 2: Encode the text
type TextFileToken = TextFile & {
    token: Uint32Array
}

const tiktokenizer = async (files: TextFile[]): Promise<TextFileToken[]> => {
    const textFileTokens: TextFileToken[] = []

    for (const file of files) {
        const token = encoding.encode(file.text)
        textFileTokens.push({
            ...file,
            token,
        })
    }
    return textFileTokens
}

// Step 3: Store the tokens in the database
const MAX_TOKENS = 500

async function splitTextToMany(text: TextFileToken): Promise<TextFileToken[]> {
    const sentences = text.text.split(". ").map((sentence) => ({
        text: sentence,
        numberTokens: encoding.encode(sentence).length,
    }))

    const chunks: TextFileToken[] = []
    let tokenSoFar = 0
    let chunk: TextFileToken | null = null

    for (const sentence of sentences) {
        const numberToken = sentence.numberTokens

        if ((tokenSoFar + numberToken) > MAX_TOKENS) {
            if (chunk) {
                chunks.push({
                    filePath: text.filePath,
                    text: chunk.text,
                    token: encoding.encode(chunk.text),
                })
            } else {
                chunks.push({
                    filePath: text.filePath,
                    text: sentence.text,
                    token: encoding.encode(sentence.text),
                })
            }
        }

        if (!chunk) {
            chunk = {
                filePath: ". " + text.filePath,
                text: sentence.text,
                token: new Uint32Array(),
            }
        } else {
            chunk.text += ". " + sentence.text
        }
        tokenSoFar += numberToken
    }

    if (chunk) {
        chunks.push(chunk)
    }
    return chunks
}

async function splitTexts(texts: TextFileToken[]): Promise<TextFileToken[]> {
    const shortened: TextFileToken[] = []

    for (const file of texts) {
        if (file.token.length > MAX_TOKENS) {
            const chunks = await splitTextToMany(file)
            shortened.push(...chunks)
        } else {
            shortened.push(file)
        }
    }
    return shortened
}


async function main() {
    const FOLDER = "nextjs"

    const texts = await cacheWithFile(() => processFiles(FOLDER), "processed/text.json")

    const textsTokens = await cacheWithFile(() => tiktokenizer(texts), "processed/textsTokens.json")
    
    const textsTokensShortened = await cacheWithFile(() => splitTexts(textsTokens), "processed/textsTokensShortened.json")
    
}


main()

async function cacheWithFile<T>(
    func: () => Promise<T>,
    filePath: string
  ): Promise<T> {
    console.log("Running function", func.toString())
    console.log("Cache file: ", filePath)
    try {
      await fs.access(filePath)
      const fileData = await fs.readFile(filePath, "utf-8")

      console.log("Using cache file")
      return JSON.parse(fileData)
    } catch {
      const data = await func()

      console.log("Writing cache file")
      await fs.writeFile(filePath, JSON.stringify(data))
      return data
    }
}