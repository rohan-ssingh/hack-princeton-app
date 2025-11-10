"use client";

import type { UseChatHelpers } from "@ai-sdk/react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react";
import { useState } from "react";
import type { Vote } from "@/lib/db/schema";
import type { Attachment, ChatMessage } from "@/lib/types";
import type { AppUsage } from "@/lib/usage";
import { Button } from "./ui/button";
import { Messages } from "./messages";
import { MultimodalInput } from "./multimodal-input";
import type { VisibilityType } from "./visibility-selector";

const CONTAINER_WIDTH = "28rem";
const MINIMIZED_WIDTH = "3rem";

type ChatSideContainerProps = {
  chatId: string;
  isReadonly: boolean;
  messages: ChatMessage[];
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  regenerate: UseChatHelpers<ChatMessage>["regenerate"];
  status: UseChatHelpers<ChatMessage>["status"];
  votes: Vote[] | undefined;
  selectedModelId: string;
  attachments: Attachment[];
  setAttachments: (attachments: Attachment[] | ((prev: Attachment[]) => Attachment[])) => void;
  input: string;
  setInput: (input: string | ((prev: string) => string)) => void;
  sendMessage: UseChatHelpers<ChatMessage>["sendMessage"];
  stop: UseChatHelpers<ChatMessage>["stop"];
  selectedVisibilityType: VisibilityType;
  onModelChange?: (modelId: string) => void;
  usage?: AppUsage;
  isArtifactVisible: boolean;
  isMinimized?: boolean;
  onMinimizedChange?: (minimized: boolean) => void;
};

export function ChatSideContainer({
  chatId,
  isReadonly,
  messages,
  setMessages,
  regenerate,
  status,
  votes,
  selectedModelId,
  attachments,
  setAttachments,
  input,
  setInput,
  sendMessage,
  stop,
  selectedVisibilityType,
  onModelChange,
  usage,
  isArtifactVisible,
  isMinimized: isMinimizedProp,
  onMinimizedChange,
}: ChatSideContainerProps) {
  const [internalMinimized, setInternalMinimized] = useState(false);
  const isMinimized = isMinimizedProp ?? internalMinimized;
  const setIsMinimized = onMinimizedChange ?? setInternalMinimized;

  return (
    <motion.div
      animate={{
        width: isMinimized ? MINIMIZED_WIDTH : CONTAINER_WIDTH,
      }}
      className="fixed right-0 top-0 z-40 flex h-dvh border-l bg-background shadow-lg"
      initial={false}
      transition={{ duration: 0.3, ease: "easeInOut" }}
    >
      {/* Toggle Button */}
      <Button
        className="absolute -left-8 top-1/2 z-50 h-8 w-8 rounded-l-md rounded-r-none border border-r-0 bg-background p-0 shadow-md transition-colors hover:bg-muted"
        onClick={() => setIsMinimized(!isMinimized)}
        type="button"
        variant="outline"
      >
        {isMinimized ? (
          <ChevronLeftIcon className="size-4" />
        ) : (
          <ChevronRightIcon className="size-4" />
        )}
        <span className="sr-only">{isMinimized ? "Expand chat" : "Minimize chat"}</span>
      </Button>

      <AnimatePresence mode="wait">
        {!isMinimized && (
          <motion.div
            animate={{ opacity: 1 }}
            className="flex h-full w-full flex-col"
            exit={{ opacity: 0 }}
            initial={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {/* Messages Area */}
            <div className="flex-1 overflow-hidden">
              <Messages
                chatId={chatId}
                isArtifactVisible={isArtifactVisible}
                isReadonly={isReadonly}
                messages={messages}
                regenerate={regenerate}
                selectedModelId={selectedModelId}
                setMessages={setMessages}
                status={status}
                votes={votes}
              />
            </div>

            {/* Input Area */}
            {!isReadonly && (
              <div className="sticky bottom-0 z-10 border-t bg-background px-2 pb-3 pt-2 md:px-4 md:pb-4">
                <MultimodalInput
                  attachments={attachments}
                  chatId={chatId}
                  input={input}
                  messages={messages}
                  onModelChange={onModelChange}
                  selectedModelId={selectedModelId}
                  selectedVisibilityType={selectedVisibilityType}
                  sendMessage={sendMessage}
                  setAttachments={setAttachments}
                  setInput={setInput}
                  setMessages={setMessages}
                  status={status}
                  stop={stop}
                  usage={usage}
                />
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Minimized State - Show Icon Only */}
      {isMinimized && (
        <motion.div
          animate={{ opacity: 1 }}
          className="flex h-full w-full items-center justify-center"
          initial={{ opacity: 0 }}
        >
          <div className="flex flex-col items-center gap-2">
            <div className="size-6 rounded-full bg-primary/10 flex items-center justify-center">
              <svg
                className="size-4 text-primary"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                viewBox="0 0 24 24"
              >
                <path
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <span className="text-[10px] text-muted-foreground">Chat</span>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

