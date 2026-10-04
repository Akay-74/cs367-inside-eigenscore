// Renders index.html to an MP4, frame by frame, so the animation is smooth
// regardless of machine speed.
//   node render.mjs                 -> out/inside_explainer.mp4
//   node render.mjs --stills 5,30   -> out/still_5.png, out/still_30.png
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, 'out');
mkdirSync(outDir, { recursive: true });
const FPS = Number(process.env.FPS || 30);

const browser = await chromium.launch(
  process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
await page.goto('file://' + path.join(here, 'index.html') + '?render=1');
await page.evaluate(() => document.fonts.ready);
const total = await page.evaluate(() => window.TOTAL);

const stillsArg = process.argv.indexOf('--stills');
if (stillsArg !== -1) {
  for (const s of process.argv[stillsArg + 1].split(',').map(Number)) {
    await page.evaluate(t => window.renderAt(t), s);
    await page.screenshot({ path: path.join(outDir, `still_${s}.png`) });
  }
  await browser.close();
  process.exit(0);
}

const out = path.join(outDir, 'inside_explainer.mp4');
const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS),
  '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
  '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });

const frames = Math.round(total * FPS);
for (let f = 0; f <= frames; f++) {
  await page.evaluate(t => window.renderAt(t), f / FPS);
  const buf = await page.screenshot({ type: 'jpeg', quality: 95 });
  if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
  if (f % (FPS * 10) === 0) console.log(`${(f / FPS).toFixed(0)}s / ${total}s`);
}
ff.stdin.end();
await new Promise(r => ff.on('close', r));
await browser.close();
console.log('wrote', out);
