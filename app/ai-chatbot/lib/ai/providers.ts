import {
  customProvider,
  extractReasoningMiddleware,
  wrapLanguageModel,
} from "ai";
import { openai } from "@ai-sdk/openai";
import { huggingFaceModel } from "./huggingface-model";

import { isTestEnvironment } from "../constants";
import {
  artifactModel,
  chatModel,
  reasoningModel,
  titleModel,
} from "./models.test";

export const myProvider = isTestEnvironment
  ? customProvider({
      languageModels: {
        "chat-model": chatModel,
        "chat-model-reasoning": huggingFaceModel,
        "title-model": titleModel,
        "artifact-model": artifactModel,
        "huggingface-model": huggingFaceModel, // Changed from "huggingface-tinyllama"
      },
    })
  : customProvider({
      languageModels: {
        "chat-model": openai("gpt-4.1-nano"),
        "chat-model-reasoning": huggingFaceModel,
        "title-model": openai("gpt-4.1-nano"),
        "artifact-model": openai("gpt-4.1-nano"),
        "huggingface-model": huggingFaceModel, // Changed from "huggingface-tinyllama"
      },
    });
