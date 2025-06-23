export const DEFAULT_CHAT_MODEL: string = "chat-model";

export interface ChatModel {
  id: string;
  name: string;
  description: string;
}

export const chatModels: Array<ChatModel> = [
  {
    id: "chat-model",
    name: "Gpt-4.1-nano-ECommerce-Chatbot",
    description: "Primary model for all-purpose chat",
  },
  {
    id: "chat-model-reasoning",
    name: "TinyLlama-ECommerce-Chatbot",
    description: "Uses advanced reasoning",
  },
  {
    id: "huggingface-model",
    name: "TinyLlama-ECommerce-Chatbot",
    description: "Custom Hugging Face Space model",
  },
];
