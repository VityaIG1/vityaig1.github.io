import 'dotenv/config';
import http from 'http';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const httpServer = http.createServer(app);
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '2mb' }));

let cachedAi: GoogleGenAI | null = null;
let cachedKey = '';
function getAiClient(): GoogleGenAI {
  const apiKey = (process.env.GEMINI_API_KEY || process.env.API_KEY || '').trim();
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured');
  }
  if (!cachedAi || cachedKey !== apiKey) {
    cachedKey = apiKey;
    cachedAi = new GoogleGenAI({
      apiKey,
      httpOptions: {
        timeout: 15000,
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return cachedAi;
}

app.get('/api/health', (_req, res) => {
  const hasApiKey = Boolean((process.env.GEMINI_API_KEY || process.env.API_KEY || '').trim());
  res.json({ ok: true, hasApiKey });
});

const RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    thoughtSteps: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: '3-4 шага размышлений именно над тем, что написал пользователь.',
    },
    reply: {
      type: Type.STRING,
      description: 'Прямой, умный и естественный ответ пользователю.',
    },
    artifactTitle: {
      type: Type.STRING,
      description: 'Название файла артефакта (например, app.js, plan.md) или пустая строка, если артефакт не нужен.',
    },
    artifactCode: {
      type: Type.STRING,
      description: 'Код или структурированный документ артефакта, либо пустая строка.',
    },
  },
  required: ['thoughtSteps', 'reply', 'artifactTitle', 'artifactCode'],
};

function resolveThinkingLevel(model: string, effort: string): ThinkingLevel | undefined {
  if (model !== 'gemini-4-hydrogen') return undefined;
  if (effort === 'Max') return ThinkingLevel.XHIGH;
  if (effort === 'High') return ThinkingLevel.HIGH;
  if (effort === 'Medium') return ThinkingLevel.MEDIUM;
  return ThinkingLevel.LOW;
}

async function callGeminiWithRetry(promptContents: string, systemInstruction: string, effort: string) {
  const attemptsConfig = [
    { model: 'gemini-3.1-flash-lite', useJsonSchema: true, timeoutMs: 12000 },
    { model: 'gemini-4-hydrogen', useJsonSchema: true, timeoutMs: 14000 },
    { model: 'gemini-3.1-flash-lite', useJsonSchema: false, timeoutMs: 12000 },
  ];
  let lastError: unknown = null;
  const ai = getAiClient();

  for (let attempt = 0; attempt < attemptsConfig.length; attempt++) {
    const { model, useJsonSchema, timeoutMs } = attemptsConfig[attempt];
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const thinkingLevel = resolveThinkingLevel(model, effort);
      const config: Record<string, unknown> = {
        systemInstruction,
        abortSignal: controller.signal,
      };
      if (useJsonSchema) {
        config.responseMimeType = 'application/json';
        config.responseSchema = RESPONSE_SCHEMA;
      }
      if (thinkingLevel) {
        config.thinkingConfig = { thinkingLevel };
      }

      const response = await ai.models.generateContent({
        model,
        contents: promptContents,
        config,
      });
      clearTimeout(timer);

      const rawText = (response.text ?? '').trim();
      if (!rawText) {
        throw new Error('Empty model response');
      }
      if (!useJsonSchema) {
        return {
          thoughtSteps: ['Анализ запроса и подготовка ответа'],
          reply: rawText,
          artifactTitle: '',
          artifactCode: '',
        };
      }
      const cleanedJson = rawText
        .replace(/^```json\s*/i, '')
        .replace(/^```\s*/i, '')
        .replace(/\s*```$/i, '')
        .trim();
      try {
        const parsed = JSON.parse(cleanedJson);
        if (parsed && typeof parsed.reply === 'string') {
          return {
            thoughtSteps: Array.isArray(parsed.thoughtSteps) ? parsed.thoughtSteps : [],
            reply: parsed.reply,
            artifactTitle: typeof parsed.artifactTitle === 'string' ? parsed.artifactTitle : '',
            artifactCode: typeof parsed.artifactCode === 'string' ? parsed.artifactCode : '',
          };
        }
        return {
          thoughtSteps: ['Анализ запроса и формулирование ответа'],
          reply: rawText,
          artifactTitle: '',
          artifactCode: '',
        };
      } catch {
        return {
          thoughtSteps: ['Анализ запроса и формулирование ответа'],
          reply: rawText,
          artifactTitle: '',
          artifactCode: '',
        };
      }
    } catch (err: unknown) {
      clearTimeout(timer);
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(`[fable-chat] attempt ${attempt + 1} (${model}, json=${useJsonSchema}) failed:`, msg);

      // Fail fast on auth/key errors since retries won't fix an invalid or missing API key
      const isAuthError =
        msg.includes('API_KEY_INVALID') ||
        msg.includes('PERMISSION_DENIED') ||
        msg.includes('GEMINI_API_KEY is not configured');
      if (isAuthError) {
        throw err;
      }

      if (attempt < attemptsConfig.length - 1) {
        await new Promise((r) => setTimeout(r, 350 * (attempt + 1)));
      }
    }
  }
  throw lastError;
}

app.post('/api/fable-chat', async (req, res) => {
  try {
    const {
      message,
      history = [],
      modelName = 'Sonnet 5.5',
      effort = 'Low',
      mode = 'chat',
      attachments = [],
    } = req.body || {};

    if (!message || typeof message !== 'string') {
      res.status(400).json({ error: 'Empty message' });
      return;
    }

    const sanitizedHistory = Array.isArray(history)
      ? history.filter(
          (h: { role?: string; text?: string }) =>
            h &&
            typeof h.text === 'string' &&
            h.text.trim().length > 0 &&
            !h.text.includes('Сейчас сервер ИИ временно не ответил')
        )
      : [];

    const historyContext =
      sanitizedHistory.length > 0
        ? '\n\nКонтекст предыдущих сообщений диалога:\n' +
          sanitizedHistory
            .slice(-10)
            .map((h: { role: string; text: string }) => `${h.role === 'user' ? 'Пользователь' : 'Claude'}: ${h.text}`)
            .join('\n')
        : '';

    const attachContext =
      Array.isArray(attachments) && attachments.length > 0
        ? `\nПрикреплённые файлы: ${attachments.join(', ')}`
        : '';

    const effortGuide =
      effort === 'Max'
        ? 'Дай максимально глубокий, всесторонний и продуманный ответ с нюансами, аргументами и примерами.'
        : effort === 'High'
          ? 'Дай подробный, хорошо структурированный и глубокий ответ.'
          : effort === 'Medium'
            ? 'Дай чёткий и содержательный ответ оптимальной длины.'
            : 'Отвечай живо, точно и лаконично, без лишней воды.';

    const systemInstruction = `Ты — Claude (${modelName}, глубина мышления: ${effort}) на тарифе "Ultra plan".
Ты универсальный, эрудированный, внимательный и приятный собеседник и помощник для любых тем: живое общение, советы, объяснение сложных идей, творчество, редактура текста, учёба, логика, программирование и дизайн.

Правила:
1. Отвечай на языке пользователя естественно, грамотно и по существу — как умный живой собеседник. ${effortGuide}
2. НИКОГДА не своди обычный разговор к HTML, программированию или «тестированию HTML», если пользователь сам не просит написать код, сайт, виджет или HTML-страницу.
3. Если пользователь действительно просит создать HTML-страницу, интерфейс, игру или прислал HTML-код на проверку — напиши чистый самодостаточный код со встроенными <style> и <script> в поле artifactCode и укажи artifactTitle с расширением ".html" (например, "index.html"), чтобы он открылся в окне Preview. Если просит код на другом языке (Python, JS, SQL и т.д.) — помести его в artifactCode с подходящим именем файла.
4. Для обычного общения, вопросов, объяснений, идей, шуток или обсуждений оставляй поля artifactTitle и artifactCode пустыми строками "".
5. Режим диалога: "${mode === 'plan' ? 'Plan (структурированный план)' : 'Chat (свободное общение)'}". ${
      mode === 'plan'
        ? 'Оформи ответ как ясный, практичный пошаговый план действий с краткими пояснениями.'
        : 'Поддерживай естественный диалог, задавай уместные уточняющие вопросы, если это помогает пользователю.'
    }
6. В поле thoughtSteps верни 2–4 коротких, осмысленных шага твоих размышлений именно над этим запросом.`;

    const parsed = await callGeminiWithRetry(
      `Сообщение пользователя: ${message}${attachContext}${historyContext}`,
      systemInstruction,
      String(effort || 'Low')
    );
    res.json(parsed);
  } catch (error: unknown) {
    const errMsg = error instanceof Error ? error.message : String(error);
    console.error('Error in /api/fable-chat:', errMsg);
    const statusCode =
      errMsg.includes('API_KEY_INVALID') || errMsg.includes('PERMISSION_DENIED')
        ? 403
        : errMsg.includes('429') || errMsg.includes('RESOURCE_EXHAUSTED')
          ? 429
          : errMsg.includes('GEMINI_API_KEY is not configured')
            ? 500
            : 503;
    res.status(statusCode).json({ error: errMsg, code: statusCode });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        ws: { server: httpServer },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
