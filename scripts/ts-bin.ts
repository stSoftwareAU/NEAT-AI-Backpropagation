/** Shared `.bin` listing / reading for the TypeScript dual-run harness. */

/** Find `.bin` files with the same order as neat-core `find_bin_files`. */
export function findBinFiles(dir: string): string[] {
  const names: string[] = [];
  for (const entry of Deno.readDirSync(dir)) {
    if (entry.isFile && entry.name.endsWith(".bin")) {
      names.push(entry.name);
    }
  }
  names.sort((a, b) => compareBinNames(a, b));
  return names.map((name) => `${dir.replace(/\/$/, "")}/${name}`);
}

function numericStem(name: string): number | undefined {
  const stem = name.replace(/\.bin$/, "");
  if (!/^[0-9]+$/.test(stem)) return undefined;
  return Number(stem);
}

function compareBinNames(a: string, b: string): number {
  const an = numericStem(a);
  const bn = numericStem(b);
  if (an !== undefined && bn !== undefined) return an - bn;
  if (an !== undefined) return -1;
  if (bn !== undefined) return 1;
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Yield each record as a Float32Array of `width` values. */
export function* readRecords(
  path: string,
  width: number,
): Generator<Float32Array> {
  const bytes = Deno.readFileSync(path);
  const bytesPerRecord = width * 4;
  if (bytes.byteLength % bytesPerRecord !== 0) {
    throw new Error(
      `${path} size ${bytes.byteLength} is not a multiple of record size ${bytesPerRecord}`,
    );
  }
  const view = new Float32Array(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength / 4,
  );
  for (let offset = 0; offset < view.length; offset += width) {
    yield view.subarray(offset, offset + width);
  }
}
