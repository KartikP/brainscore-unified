// Refresh the distributable bundle after a developer builds the frontend.
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const destination = path.join(root, '../backend/umi_lab/static');
const digest = async filename => createHash('sha256').update((await readFile(filename, 'utf8')).replace(/\r\n/g, '\n')).digest('hex');
const sources = [...['index.html', 'style.css', 'package.json', 'package-lock.json', 'tsconfig.json', 'build.mjs', 'package.mjs'],
  ...(await readdir(path.join(root, 'src'))).filter(name => name.endsWith('.ts')).map(name => `src/${name}`)].sort();
const assets = (await readdir(path.join(root, 'dist'))).filter(name => /\.(js|html|css)$/.test(name)).sort();
if (!assets.includes('index.html') || !assets.includes('main.js')) throw new Error('Build the frontend before packaging it');
await mkdir(destination, { recursive: true });
for (const name of await readdir(destination)) await rm(path.join(destination, name), { force: true });
const manifest = { version: 1, sources: {}, assets: {} };
for (const name of sources) manifest.sources[name] = await digest(path.join(root, name));
for (const name of assets) {
  await copyFile(path.join(root, 'dist', name), path.join(destination, name));
  manifest.assets[name] = await digest(path.join(destination, name));
}
await writeFile(path.join(destination, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`Packaged ${assets.length} frontend files for Python installation`);
