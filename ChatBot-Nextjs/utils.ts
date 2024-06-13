import { Logger } from "tslog";

const isDev = process.env.NODE_ENV === 'development';

export const logger = new Logger({
    stylePrettyLogs: isDev,
    minLevel: isDev ? 0 : 3,
})
