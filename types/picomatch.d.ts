declare module "picomatch" {
  export type Matcher = (input: string) => boolean;

  export type PicomatchOptions = {
    dot?: boolean;
  };

  export default function picomatch(
    pattern: string | readonly string[],
    options?: PicomatchOptions
  ): Matcher;
}
