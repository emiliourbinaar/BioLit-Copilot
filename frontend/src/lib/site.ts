/** Where the records this site quotes live. Links point at `master`, the published history. */
export const REPO_URL = "https://github.com/emiliourbinaar/BioLit-Copilot";

export function docUrl(file: string): string {
  return `${REPO_URL}/blob/master/docs/${file}`;
}
