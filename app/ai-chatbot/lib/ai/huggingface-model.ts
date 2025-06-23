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

      const response = await fetch(
        "https://shenghaoyummy-ai-chatbot.hf.space/gradio_api/run/chat",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            data: [userMessage, []],
          }),
        }
      );

      if (!response.ok) {
        console.error(
          "❌ HuggingFace API error:",
          response.status,
          response.statusText
        );

        // Return a fallback response instead of throwing
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

      // Return fallback instead of throwing
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
      throw error;
    }
  },
};
