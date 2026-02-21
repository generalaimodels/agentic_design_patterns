import { buildLegacyPaths, buildSlugFromSourcePath, slugifySegment } from "@/content/utils";

describe("content utils", () => {
  it("slugifies path segments deterministically", () => {
    expect(slugifySegment("Inter-Agent Communication (A2A)")).toBe("inter-agent-communication-a2a");
    expect(buildSlugFromSourcePath("Inter-Agent Communication (A2A)/inter_agent_communication.md", "Fallback")).toEqual([
      "inter-agent-communication-a2a",
      "inter-agent-communication"
    ]);
  });

  it("builds legacy encoded paths for redirect handling", () => {
    expect(buildLegacyPaths("Exception Handling and Recovery/exception_handling_and_recovery.md")).toEqual([
      "/docs/Exception%20Handling%20and%20Recovery/exception_handling_and_recovery"
    ]);
  });
});
