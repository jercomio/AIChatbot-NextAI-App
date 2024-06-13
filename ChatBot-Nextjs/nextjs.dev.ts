import { chromium } from "playwright"
import fs from 'fs/promises'

const run = async () => {
    const browser = await chromium.launch()
    const page = await browser.newPage()
    await page.goto('https://nextjs.org/docs')

    const nav = await page.$('nav.styled-scrollbar')
    
    if (!nav) {
        throw new Error('nav not found')
    }

    const links = await nav.$$('a')

    const urls = await Promise.all(
        links.map(async (link) => {
            const href = await link.getAttribute('href')
            return href
        })
    )
    for (const url of urls) {
        await page.goto(`https://nextjs.org${url}`)

        const content = await page.$eval(
            'div.prose.prose-vercel',
            el => el.textContent
        )
    
        if (!content) {
            continue
        }
        
        const encodedURLForFileName = `https://nextjs.org${url}`.replace(/\//g, '_')
    
        const filePath = `./data/nextjs/${encodedURLForFileName}.txt`
    
        fs.writeFile(filePath, content)
    }

    await browser.close()
}
run()
