import { describe, expect, it } from "vitest";
import { clusterDomId } from "../src/lib/dom";
import { RUNS } from "../src/lib/fixtures";

describe("clusterDomId", () => {
  it("turns a cluster key into a fragment-safe id that stays unique within every run", () => {
    expect(clusterDomId("MESH:D002945|MESH:D007674")).toBe("cluster-MESH-D002945-MESH-D007674");
    for (const run of RUNS) {
      const ids = run.clusters.map((c) => clusterDomId(c.key));
      expect(new Set(ids).size, run.slug).toBe(ids.length);
      for (const id of ids) expect(id).toMatch(/^[A-Za-z0-9-]+$/);
    }
  });
});
