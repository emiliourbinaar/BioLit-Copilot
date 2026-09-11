/** `MESH:D002945|MESH:D007674` -> `cluster-MESH-D002945-MESH-D007674`, safe in a URL fragment. */
export function clusterDomId(key: string): string {
  return `cluster-${key.replace(/[^A-Za-z0-9]+/g, "-")}`;
}
