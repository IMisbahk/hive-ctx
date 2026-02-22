import fs from "node:fs";
import path from "node:path";

type NativeModule = {
  HiveCtxEngine: new (storagePath: string, budgetTokens?: number) => {
    storagePath: string;
    budgetTokens?: number;
    classifyMessage(message: string): ClassifierResultDto;
    pipelineBuild(
      message: string,
      user_profile: Record<string, string>,
      token_budget?: number | null,
    ): PipelineResultDto;
    graphAddNode(text: string, category?: string | null): Array<{ id: number }>;
    memoryStore(text: string): number;
  };
};

function loadNative(): NativeModule {
  const envPath = process.env.HIVE_CTX_NATIVE_PATH;
  if (envPath) {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require(envPath) as NativeModule;
  }

  const addonDir = path.join(__dirname, "..");
  const directPath = path.join(addonDir, "hive_ctx.node");
  if (fs.existsSync(directPath)) {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require(directPath) as NativeModule;
  }

  const namedCandidates = fs
    .readdirSync(addonDir)
    .filter((f) => f.startsWith("hive_ctx.") && f.endsWith(".node"))
    .sort();

  if (namedCandidates.length > 0) {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require(path.join(addonDir, namedCandidates[0])) as NativeModule;
  }

  const anyCandidate = fs
    .readdirSync(addonDir)
    .filter((f) => f.endsWith(".node"))
    .sort();

  if (anyCandidate.length > 0) {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require(path.join(addonDir, anyCandidate[0])) as NativeModule;
  }

  throw new Error(
    `Failed to load native addon. Build it with "npm run build:native" (expected hive_ctx*.node in ${addonDir}).`,
  );
}

const native = loadNative();

export type ClassifierResultDto = {
  temporalWeight: number;
  personalWeight: number;
  technicalWeight: number;
  emotionalWeight: number;
  messageType: string;
  sessionState: string;
};

export type PipelineLayersDto = {
  episodes: number;
  graphNodes: number;
  fingerprintEntries: number;
  fingerprintMode: string;
  includedLayers: string[];
};

export type PipelineResultDto = {
  systemPrompt: string;
  tokenCount: number;
  layers: PipelineLayersDto;
};

export interface HiveCtxConfig {
  storagePath: string;
  budgetTokens?: number;
  model?: string;
  profile?: Record<string, string>;
}

export type PluginRetrieveResult = {
  content: string;
  tokens: number;
};

export interface HiveCtxPlugin {
  name: string;
  retrieve(
    message: string,
    weights: ClassifierResultDto,
  ): Promise<PluginRetrieveResult>;
}

export interface PluginContribution {
  name: string;
  content: string;
  tokens: number;
}

export interface ContextResult {
  systemPrompt: string;
  tokenCount: number;
  fingerprintMode: string;
  layers: string[];
  pluginContributions: PluginContribution[];
}

const DEFAULT_TOKEN_BUDGET = 300;

export class HiveCtx {
  private readonly inner: InstanceType<NativeModule["HiveCtxEngine"]>;
  private readonly plugins = new Map<string, HiveCtxPlugin>();
  private readonly profile: Record<string, string>;
  private readonly budgetTokens?: number;

  constructor(private readonly config: HiveCtxConfig) {
    this.profile = config.profile ?? {};
    this.budgetTokens = config.budgetTokens;
    this.inner = new native.HiveCtxEngine(config.storagePath, config.budgetTokens);
  }

  public async build(message: string): Promise<ContextResult> {
    const classified = this.inner.classifyMessage(message);
    const pipeline = this.inner.pipelineBuild(
      message,
      this.profile,
      this.budgetTokens ?? null,
    );
    const budgetLimit = this.budgetTokens ?? DEFAULT_TOKEN_BUDGET;
    const pluginBudget = Math.max(0, budgetLimit - pipeline.tokenCount);
    const pluginContext = await this.runPlugins(message, classified, pluginBudget);

    let prompt = pipeline.systemPrompt;
    if (pluginContext.contributions.length > 0) {
      prompt +=
        "\n\n" +
        pluginContext.contributions
          .map((contribution) => `[${contribution.name}] ${contribution.content}`)
          .join("\n");
    }

    return {
      systemPrompt: prompt,
      tokenCount: pipeline.tokenCount + pluginContext.tokensUsed,
      fingerprintMode: pipeline.layers.fingerprintMode,
      layers: pipeline.layers.includedLayers,
      pluginContributions: pluginContext.contributions,
    };
  }

  public async remember(fact: string): Promise<void> {
    const sanitized = fact.trim();
    if (sanitized.length === 0) {
      return;
    }
    this.inner.graphAddNode(sanitized);
  }

  public async episode(message: string, response: string): Promise<void> {
    const trimmedMessage = message.trim();
    const trimmedResponse = response.trim();
    if (!trimmedMessage && !trimmedResponse) {
      return;
    }
    const payload = [trimmedMessage, trimmedResponse].filter(Boolean).join(" || ");
    this.inner.memoryStore(payload);
  }

  public use(plugin: HiveCtxPlugin): void {
    this.plugins.set(plugin.name, plugin);
  }

  private async runPlugins(
    message: string,
    weights: ClassifierResultDto,
    budget: number,
  ): Promise<{ contributions: PluginContribution[]; tokensUsed: number }> {
    const contributions: PluginContribution[] = [];
    let tokensUsed = 0;
    let remaining = budget;

    for (const plugin of this.plugins.values()) {
      if (remaining <= 0) {
        break;
      }
      const candidate = await plugin.retrieve(message, weights);
      const snippet = candidate.content.trim();
      if (!snippet || candidate.tokens <= 0 || candidate.tokens > remaining) {
        continue;
      }
      contributions.push({
        name: plugin.name,
        content: snippet,
        tokens: candidate.tokens,
      });
      tokensUsed += candidate.tokens;
      remaining -= candidate.tokens;
    }

    return { contributions, tokensUsed };
  }
}
