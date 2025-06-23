import { LanguageModelV1 } from "ai";

function transformPromptToGradio(prompt: any): string {
  console.log("🔍 Full prompt object:", JSON.stringify(prompt, null, 2));

  // Handle case where prompt is an array of messages directly
  let messages = prompt;

  // If prompt has a messages property, use that
  if (prompt.messages && Array.isArray(prompt.messages)) {
    messages = prompt.messages;
  }

  // If prompt is not an array, return empty
  if (!Array.isArray(messages)) {
    console.warn("⚠️ Prompt is not an array of messages");
    return prompt.text || "";
  }

  // Get the last user message
  const lastMessage = messages[messages.length - 1];
  console.log("📝 Last message:", lastMessage);

  if (lastMessage && lastMessage.role === "user" && lastMessage.content) {
    // Handle different content formats
    if (typeof lastMessage.content === "string") {
      console.log("📝 Extracted string content:", lastMessage.content);
      return lastMessage.content;
    } else if (Array.isArray(lastMessage.content)) {
      // Extract text from content array
      const textContent = lastMessage.content
        .filter((part: any) => part.type === "text")
        .map((part: any) => part.text)
        .join(" ");
      console.log("📝 Extracted array content:", textContent);
      return textContent;
    }
  }

  // Fallback to text property
  console.warn("⚠️ Could not extract user message, using fallback");
  return prompt.text || "";
}

function extractTextFromGradioResponse(result: any): string {
  return result.data?.[0] || "";
}

// Helper function to detect timeout-related errors
function isTimeoutError(error: any): boolean {
  const errorMessage = error?.message?.toLowerCase() || "";
  const errorName = error?.name?.toLowerCase() || "";
  const errorCode = error?.code || "";

  return (
    errorMessage.includes("timeout") ||
    errorMessage.includes("timed out") ||
    errorMessage.includes("network timeout") ||
    errorMessage.includes("connection timeout") ||
    errorName.includes("timeout") ||
    errorCode === "TIMEOUT" ||
    errorCode === "ETIMEDOUT" ||
    errorCode === "ECONNRESET" ||
    errorCode === "ENOTFOUND" ||
    // Specific to fetch timeouts
    errorMessage.includes("fetch") ||
    errorMessage.includes("aborted") ||
    errorMessage.includes("network error")
  );
}

export const huggingFaceModel: LanguageModelV1 = {
  specificationVersion: "v1",
  provider: "huggingface",
  modelId: "tinyllama-ecommerce",
  defaultObjectGenerationMode: "json",

  doGenerate: async ({ prompt, mode, ...options }) => {
    console.log(
      "🤖 HuggingFace model called with prompt:",
      transformPromptToGradio(prompt)
    );

    try {
      const userMessage = transformPromptToGradio(prompt);

      // Don't send empty messages
      if (!userMessage.trim()) {
        console.warn("⚠️ Empty message, using fallback");
        return {
          rawCall: { rawPrompt: prompt, rawSettings: options },
          finishReason: "stop",
          usage: { promptTokens: 0, completionTokens: 0 },
          text: "I'm sorry, I didn't receive your message. Could you please try again?",
        };
      }

      // Add timeout to the fetch request
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch(
        "https://shenghaoyummy-ai-chatbot.hf.space/gradio_api/run/chat",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            data: [userMessage, []],
          }),
          signal: controller.signal,
        }
      );

      clearTimeout(timeoutId);

      if (!response.ok) {
        console.error(
          "❌ HuggingFace API error:",
          response.status,
          response.statusText
        );

        // Check if it's a 502, 503, 504 (server unavailable/timeout related)
        if ([502, 503, 504].includes(response.status)) {
          return {
            rawCall: { rawPrompt: prompt, rawSettings: options },
            finishReason: "stop",
            usage: { promptTokens: 0, completionTokens: 0 },
            text: "Please contact author to enable Hugging Face space server for Fine-tuned TinyLlama model demo or you can try to use the chatgpt model at the top left of the page.",
          };
        }

        // Return a fallback response for other errors
        return {
          rawCall: { rawPrompt: prompt, rawSettings: options },
          finishReason: "stop",
          usage: { promptTokens: 0, completionTokens: 0 },
          text: "I'm experiencing technical difficulties. Please try again in a moment.",
        };
      }

      const result = await response.json();
      console.log("✅ HuggingFace API response:", result);

      const text = extractTextFromGradioResponse(result);
      console.log("📝 Extracted text:", text);

      return {
        rawCall: { rawPrompt: prompt, rawSettings: options },
        finishReason: "stop",
        usage: { promptTokens: 0, completionTokens: 0 },
        text:
          text ||
          "I'm sorry, I couldn't generate a response. Please try again.",
      };
    } catch (error) {
      console.error("❌ HuggingFace model error:", error);

      // Check if it's a timeout-related error
      if (isTimeoutError(error)) {
        console.log("⏰ Detected timeout error, returning custom message");
        return {
          rawCall: { rawPrompt: prompt, rawSettings: options },
          finishReason: "stop",
          usage: { promptTokens: 0, completionTokens: 0 },
          text: "Please contact author to enable Hugging Face space server for demo.",
        };
      }

      // Return fallback for other errors
      return {
        rawCall: { rawPrompt: prompt, rawSettings: options },
        finishReason: "stop",
        usage: { promptTokens: 0, completionTokens: 0 },
        text: "I'm currently unavailable. Please try again later.",
      };
    }
  },

  doStream: async ({ prompt, mode, ...options }) => {
    console.log("🌊 HuggingFace streaming called");

    try {
      const result = await huggingFaceModel.doGenerate({
        prompt,
        mode,
        ...options,
      });

      const stream = new ReadableStream({
        start(controller) {
          console.log("📡 Starting stream with text:", result.text);
          controller.enqueue({ type: "text-delta", textDelta: result.text });
          controller.enqueue({
            type: "finish",
            finishReason: "stop",
            usage: result.usage,
          });
          controller.close();
        },
      });

      return {
        stream,
        rawCall: { rawPrompt: prompt, rawSettings: options },
      };
    } catch (error) {
      console.error("❌ HuggingFace streaming error:", error);

      // For streaming, we'll create an error stream
      if (isTimeoutError(error)) {
        const errorStream = new ReadableStream({
          start(controller) {
            controller.enqueue({
              type: "text-delta",
              textDelta:
                "Please contact author to enable Hugging Face space server for demo.",
            });
            controller.enqueue({
              type: "finish",
              finishReason: "stop",
              usage: { promptTokens: 0, completionTokens: 0 },
            });
            controller.close();
          },
        });

        return {
          stream: errorStream,
          rawCall: { rawPrompt: prompt, rawSettings: options },
        };
      }

      throw error;
    }
  },
};
