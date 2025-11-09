"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { signOut, useSession } from "next-auth/react";
import { useTheme } from "next-themes";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ChevronDown,
  ChevronRight,
  Loader2,
  LogIn,
  LogOut,
  Send,
  Sparkles,
} from "lucide-react";
import { createArticleSlug, getArticles } from "@/lib/articles";
import type { ChatMessage } from "@/lib/types";
import type { AppUsage } from "@/lib/usage";
import type { VisibilityType } from "./visibility-selector";
import { cn, generateUUID } from "@/lib/utils";
import { toast } from "./toast";

const normalizeCategory = (value: string) => value.trim().toLowerCase();

type ChatProps = {
  id: string;
  initialMessages: ChatMessage[];
  initialChatModel: string;
  initialVisibilityType: VisibilityType;
  isReadonly: boolean;
  autoResume: boolean;
  initialLastContext?: AppUsage;
  initialUserTopics?: string[];
};

type MessagePart = ChatMessage["parts"][number];

const isTextPart = (
  part: MessagePart
): part is MessagePart & { type: "text"; text: string } =>
  part.type === "text" && typeof (part as { text?: unknown }).text === "string";

const isCitationPart = (
  part: MessagePart
): part is MessagePart & { type: "citation"; text: string } =>
  part.type === "citation" &&
  typeof (part as { text?: unknown }).text === "string";

const normalizeBackendEndpoint = () => {
  const base =
    (process.env.NEXT_PUBLIC_BACKEND_URL ??
      "http://localhost:8000") as string;
  return `${base.replace(/\/+$/, "")}/user-query`;
};

const getCitationLabel = (url: string) => {
  try {
    const { hostname } = new URL(url);
    return hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
};

type ConversationMessagePayload = {
  role: "system" | "user" | "assistant";
  content: string;
};

const getTextFromChatMessage = (message: ChatMessage): string => {
  return message.parts
    .filter(isTextPart)
    .map((part) => part.text.trim())
    .filter(Boolean)
    .join("\n")
    .trim();
};

const buildConversationPayload = (messages: ChatMessage[]): ConversationMessagePayload[] => {
  return messages
    .map((message) => {
      const content = getTextFromChatMessage(message);
      if (!content) {
        return null;
      }

      const normalizedRole:
        | ConversationMessagePayload["role"]
        = message.role === "assistant"
          ? "assistant"
          : message.role === "system"
          ? "system"
          : "user";

      return {
        role: normalizedRole,
        content,
      };
    })
    .filter(Boolean) as ConversationMessagePayload[];
};

const extractUrlFromMetadata = (metadata: Record<string, any> | undefined): string | null => {
  if (!metadata || typeof metadata !== "object") {
    return null;
  }

  const candidateChecks: Array<[unknown, (value: string) => boolean]> = [
    [metadata.url, () => true],
    [metadata.source_url, () => true],
    [metadata.sourceUrl, () => true],
    [metadata.source, () => true],
    [metadata.file_name, (value: string) => /^https?:\/\//i.test(value)],
    [metadata.fileName, (value: string) => /^https?:\/\//i.test(value)],
  ];

  for (const [candidate, validator] of candidateChecks) {
    if (typeof candidate === "string") {
      const trimmed = candidate.trim();
      if (trimmed && validator(trimmed)) {
        return trimmed;
      }
    }
  }

  return null;
};

export function Chat({
  id,
  initialMessages,
  initialChatModel: _initialChatModel,
  initialVisibilityType: _initialVisibilityType,
  isReadonly,
  autoResume: _autoResume,
  initialLastContext: _initialLastContext,
  initialUserTopics = [],
}: ChatProps) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { data: session, status: sessionStatus } = useSession();
  const { resolvedTheme, setTheme } = useTheme();

  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [hasAppendedQuery, setHasAppendedQuery] = useState(false);
  const [isChatExpanded, setIsChatExpanded] = useState(true);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const [selectedText, setSelectedText] = useState("");
  const [chatStatus, setChatStatus] = useState<"idle" | "loading">("idle");
  const [selectedFeedTopic, setSelectedFeedTopic] = useState<string>("All");

  const messageEndRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const userMenuRef = useRef<HTMLDivElement | null>(null);
  const mainFeedRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    setMessages(initialMessages);
  }, [initialMessages]);

  useEffect(() => {
    setIsUserMenuOpen(false);
  }, [session?.user]);

  useEffect(() => {
    if (!isUserMenuOpen) {
      return;
    }

    const handleClickOutside = (event: MouseEvent) => {
      if (
        userMenuRef.current &&
        !userMenuRef.current.contains(event.target as Node)
      ) {
        setIsUserMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isUserMenuOpen]);

  useEffect(() => {
    const handleSelectionChange = () => {
      const mainElement = mainFeedRef.current;
      if (!mainElement) {
        return;
      }

      const selection = window.getSelection();
      if (!selection || selection.isCollapsed) {
        setSelectedText("");
        return;
      }

      const anchorNode = selection.anchorNode;
      const focusNode = selection.focusNode;
      const text = selection.toString().trim();

      if (
        text &&
        anchorNode &&
        focusNode &&
        mainElement.contains(anchorNode) &&
        mainElement.contains(focusNode)
      ) {
        setSelectedText(text);
        return;
      }

      setSelectedText("");
    };

    document.addEventListener("selectionchange", handleSelectionChange);
    return () => {
      document.removeEventListener("selectionchange", handleSelectionChange);
    };
  }, []);

  const backendEndpoint = useMemo(() => normalizeBackendEndpoint(), []);

  const userTopics = useMemo(() => {
    const topicsSource =
      session?.user?.topics && session.user.topics.length > 0
        ? session.user.topics
        : initialUserTopics;

    return (topicsSource ?? [])
      .map((topic) => topic.trim())
      .filter((topic) => topic.length > 0);
  }, [initialUserTopics, session?.user?.topics]);

  const categorizedArticles = useMemo(() => {
    const articles = getArticles();
    const grouped = new Map<
      string,
      { name: string; normalized: string; articles: typeof articles }
    >();

    for (const article of articles) {
      const normalized = normalizeCategory(article.categoryName);
      if (!grouped.has(normalized)) {
        grouped.set(normalized, {
          name: article.categoryName,
          normalized,
          articles: [],
        });
      }

      grouped.get(normalized)!.articles.push(article);
    }

    return Array.from(grouped.values());
  }, []);

  const preferredCategorySet = useMemo(() => {
    return new Set(userTopics.map((topic) => normalizeCategory(topic)));
  }, [userTopics]);

  const feedCategories = useMemo(() => {
    if (preferredCategorySet.size === 0) {
      return [];
    }

    return categorizedArticles.filter((category) =>
      preferredCategorySet.has(category.normalized)
    );
  }, [categorizedArticles, preferredCategorySet]);

  const availableTopicFilters = useMemo(() => {
    if (userTopics.length === 0) {
      return ["All"];
    }

    return ["All", ...userTopics];
  }, [userTopics]);

  useEffect(() => {
    if (
      selectedFeedTopic !== "All" &&
      !userTopics.some(
        (topic) => normalizeCategory(topic) === normalizeCategory(selectedFeedTopic)
      )
    ) {
      setSelectedFeedTopic("All");
    }
  }, [selectedFeedTopic, userTopics]);

  const feedArticles = useMemo(() => {
    if (feedCategories.length === 0) {
      return [];
    }

    const flattened = feedCategories.flatMap((category) =>
      category.articles.map((article) => ({
        categoryName: category.name,
        article,
        slug: createArticleSlug(category.name, article.title),
      }))
    );

    if (selectedFeedTopic === "All") {
      return flattened;
    }

    const normalizedSelection = normalizeCategory(selectedFeedTopic);
    return flattened.filter(
      (item) => normalizeCategory(item.categoryName) === normalizedSelection
    );
  }, [feedCategories, selectedFeedTopic]);

  const firstName = useMemo(() => {
    if (session?.user?.name) {
      return session.user.name.split(" ")[0];
    }
    if (session?.user?.email) {
      return (session.user.email.split("@")[0] ?? "").trim() || "there";
    }
    return "there";
  }, [session?.user?.name, session?.user?.email]);

  const userInitial = useMemo(() => {
    if (session?.user?.name) {
      return session.user.name.charAt(0).toUpperCase();
    }
    if (session?.user?.email) {
      return session.user.email.charAt(0).toUpperCase();
    }
    return undefined;
  }, [session?.user?.name, session?.user?.email]);

  const currentTheme = resolvedTheme === "dark" ? "dark" : "light";
  const nextThemeLabel = currentTheme === "dark" ? "Light" : "Dark";
  const handleToggleTheme = useCallback(() => {
    setTheme(currentTheme === "dark" ? "light" : "dark");
    setIsUserMenuOpen(false);
  }, [currentTheme, setTheme]);

  const handleNewChat = useCallback(() => {
    setIsUserMenuOpen(false);
    router.push("/");
    router.refresh();
  }, [router]);

  const handleSignOut = useCallback(() => {
    setIsUserMenuOpen(false);
    signOut({
      redirectTo: "/login",
    });
  }, [signOut]);

  const sendMessage = useCallback(
    async (message: string) => {
      const trimmed = message.trim();

      if (!trimmed) {
        return;
      }

      const conversationPayload = buildConversationPayload(messages);

      const userMessage: ChatMessage = {
        id: generateUUID(),
        role: "user",
        parts: [{ type: "text", text: trimmed }],
        metadata: { createdAt: new Date().toISOString() },
      };

      setMessages((prev) => [...prev, userMessage]);

      setChatStatus("loading");

      try {
        const response = await fetch(backendEndpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            user_query: trimmed,
            conversation: conversationPayload,
          }),
        });

        if (!response.ok) {
          throw new Error(`Backend responded with status ${response.status}`);
        }

        const data = await response.json();
        const textResponse =
          typeof data?.text_response === "string"
            ? data.text_response
            : "I'm not sure how to respond to that yet.";

        const documents = Array.isArray(data?.documents)
          ? data.documents
          : [];

        const citationParts = documents
          .map((doc: unknown) => {
            if (!doc || typeof doc !== "object") {
              return null;
            }

            const metadata = (doc as { metadata?: Record<string, any> }).metadata;
            const url = extractUrlFromMetadata(metadata);

            if (!url) {
              return null;
            }

            return { type: "citation" as const, text: url };
          })
          .filter((value): value is { type: "citation"; text: string } => Boolean(value));

        const assistantMessage: ChatMessage = {
          id: generateUUID(),
          role: "assistant",
          parts: [
            { type: "text", text: textResponse },
            ...citationParts,
          ],
          metadata: { createdAt: new Date().toISOString() },
        };

        setMessages((prev) => [...prev, assistantMessage]);
      } catch (error) {
        console.error(error);
        toast({
          type: "error",
          description:
            "We couldn't reach the chat service. Please try again.",
        });
      } finally {
        setChatStatus("idle");
      }
    },
    [backendEndpoint, messages]
  );

  const query = searchParams.get("query");

  useEffect(() => {
    if (query && !hasAppendedQuery) {
      setHasAppendedQuery(true);
      setInput(query);
      sendMessage(query);

      if (typeof window !== "undefined") {
        window.history.replaceState({}, "", `/chat/${id}`);
      }
    }
  }, [query, hasAppendedQuery, id, sendMessage]);

  useEffect(() => {
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, chatStatus]);

  const handleSend = () => {
    if (isReadonly || chatStatus === "loading") {
      return;
    }

    const trimmed = input.trim();

    if (!trimmed) {
      return;
    }

    setInput("");
    sendMessage(trimmed);
  };

  const handleInsertSelectedText = useCallback(() => {
    const trimmedSelection = selectedText.trim();
    if (!trimmedSelection) {
      return;
    }

    setInput((previous) => {
      if (!previous) {
        return trimmedSelection;
      }
      const needsSpace = /\s$/.test(previous);
      return `${previous}${needsSpace ? "" : " "}${trimmedSelection}`;
    });
    setSelectedText("");
    requestAnimationFrame(() => {
      inputRef.current?.focus();
    });
    if (typeof window !== "undefined") {
      const selection = window.getSelection();
      selection?.removeAllRanges();
    }
  }, [selectedText]);

  const isAuthLoading = sessionStatus === "loading";
  const isSendDisabled = chatStatus === "loading" || isReadonly;

  return (
    <div className="relative flex h-dvh w-full overflow-hidden bg-black text-white">
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <div className="fixed left-4 top-2 z-50">
        {isAuthLoading ? (
          <div className="flex items-center space-x-3 rounded-full border border-gray-700/40 bg-gray-900/80 px-4 text-sm text-gray-400 shadow-lg">
            <Loader2 className="h-4 w-4 animate-spin" />
            Checking session…
          </div>
        ) : session?.user ? (
          <div className="relative" ref={userMenuRef}>
            <button
              className="flex items-center space-x-3 rounded-full border border-gray-700/40 bg-gray-900/80 px-4 py-2 text-sm text-white shadow-lg transition-all hover:shadow-xl"
              onClick={() => setIsUserMenuOpen((prev) => !prev)}
              type="button"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gray-600 font-semibold uppercase">
                {userInitial}
              </div>
              <span className="font-medium">
                {session.user.name ?? session.user.email}
              </span>
              <ChevronDown className="h-4 w-4 opacity-80" />
            </button>

            {isUserMenuOpen && (
              <div className="flex w-60 bg-gray-950/95 p-3 text-sm text-gray-200 shadow-2xl backdrop-blur-xl">
                <p className="mb-2 text-xs uppercase tracking-wide text-gray-400">
                  Signed in as{" "}
                  <span className="font-semibold text-white">
                    {session.user.email ?? session.user.name}
                  </span>
                </p>
                <div className="space-y-1">
                  <button
                    className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-red-300 transition-colors hover:bg-red-500/10 hover:text-red-200"
                    onClick={handleSignOut}
                    type="button"
                  >
                    Sign out
                    <LogOut className="h-4 w-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center space-x-3">
            <button
              className="flex items-center space-x-2 rounded-full border border-gray-700/40 bg-gray-900/80 px-4 py-2 text-sm font-medium text-white shadow-lg transition-all hover:shadow-xl"
              disabled={sessionStatus === "loading"}
              onClick={() => {
                if (sessionStatus === "loading") {
                  toast({
                    type: "error",
                    description:
                      "Checking authentication status, please try again!",
                  });
                  return;
                }

                router.push("/login");
              }}
              type="button"
            >
              <LogIn className="h-4 w-4" />
              <span>Sign In</span>
            </button>
          </div>
        )}
      </div>
        <header className="border-b border-gray-800 bg-gray-950 px-6 py-4 shadow-lg">
          <div className="flex items-center justify-center space-x-3">
            <div>
              <h1 className="text-xl font-bold text-white">
                Relay
              </h1>
            </div>
          </div>
        </header>

        <main
          ref={mainFeedRef}
          className="flex-1 overflow-y-auto bg-gradient-to-br from-black via-gray-950/10 to-black px-6 py-6"
        >
          <div className="mx-auto max-w-4xl">
            <div className="mb-8">
              <div className="mb-3 flex items-center space-x-3">
                <Sparkles className="h-7 w-7 text-gray-400" />
                <h2 className="text-4xl font-bold text-white">
                  Hey {firstName}!
                </h2>
              </div>
              <p className="ml-10 text-lg text-gray-300">
                Here are some interesting stories for you today
              </p>
            </div>

            {userTopics.length > 0 && (
              <div className="mb-6 flex flex-wrap gap-2">
                {availableTopicFilters.map((topic) => (
                  <button
                    key={topic}
                    className={cn(
                      "rounded-full border px-4 py-2 text-sm font-semibold transition-all",
                      selectedFeedTopic === topic
                        ? "border-white bg-white text-black shadow-lg"
                        : "border-gray-700/60 bg-gray-800/60 text-gray-300 hover:bg-gray-700/70"
                    )}
                    onClick={() => setSelectedFeedTopic(topic)}
                    type="button"
                  >
                    {topic}
                  </button>
                ))}
              </div>
            )}

            <div className="space-y-4">
              {feedArticles.length === 0 ? (
                <div className="rounded-2xl border border-gray-800 bg-gray-900/70 p-6 text-sm text-gray-300">
                  No articles match your selected topics yet. Try adjusting your interests to discover more content.
                </div>
              ) : (
                feedArticles.map(({ article, categoryName, slug }) => (
                  <Link
                    key={slug}
                    href={`/articles/${slug}`}
                    className="block rounded-2xl border border-gray-800 bg-gray-900/60 p-5 transition-all hover:border-gray-600 hover:bg-gray-900/80"
                  >
                    <span className="inline-flex rounded-full border border-gray-700/60 bg-gray-800/70 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-gray-300">
                      {categoryName}
                    </span>
                    <h4 className="mt-3 text-lg font-semibold text-white">
                      {article.title}
                    </h4>
                    <p className="mt-2 line-clamp-3 text-sm text-gray-300">
                      {article.summary}
                    </p>
                    <span className="mt-4 inline-flex items-center text-sm font-semibold text-gray-400">
                      Read full briefing →
                    </span>
                  </Link>
                ))
              )}
            </div>
          </div>
        </main>
      </div>

      <div
        id="relay-chat-panel"
        className={cn(
          "relative z-30 flex h-full flex-col overflow-hidden border-l border-gray-800 bg-gradient-to-b from-gray-950/95 via-black to-black/95 backdrop-blur-xl shadow-2xl transition-all duration-300 ease-in-out",
          isChatExpanded
            ? "w-96 pointer-events-auto"
            : "w-0 pointer-events-none"
        )}
        aria-hidden={!isChatExpanded}
      >
        {isChatExpanded && (
          <div className="flex h-full flex-col">
            <div className="border-b border-gray-800 bg-gray-900/50 px-5 py-4">
              <div className="mb-1 flex items-center space-x-2">
                <Sparkles className="h-5 w-5 text-gray-300" />
                <h3 className="font-bold text-white">Relay</h3>
              </div>
              <p className="text-xs text-gray-400">
                Your AI transparency assistant
              </p>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4">
              <div className="flex flex-col space-y-3">
                {messages.map((message) => {
                  const textContent = message.parts
                    .filter(isTextPart)
                    .map((part) => part.text)
                    .join("\n")
                    .trim();

                  const citations = message.parts
                    .filter(isCitationPart)
                    .map((part) => part.text);

                  return (
                    <div
                      key={message.id}
                      className={cn(
                        "flex",
                        message.role === "user"
                          ? "justify-end"
                          : "justify-start"
                      )}
                    >
                      <div
                        className={cn(
                          "max-w-[85%] rounded-xl px-3 py-2.5 text-sm leading-relaxed shadow-lg",
                          message.role === "user"
                            ? "bg-white text-black"
                            : "border border-gray-700 bg-gray-800/70 text-gray-100"
                        )}
                      >
                        {textContent && (
                          <div className="space-y-2">
                            {textContent.split("\n").map((line, index) => (
                              <p key={`${message.id}-line-${index}`}>{line}</p>
                            ))}
                          </div>
                        )}

                        {citations.length > 0 && (
                          <div className="mt-3 space-y-1 border-t border-gray-700/60 pt-2 text-xs">
                            {citations.map((url, index) => (
                              <a
                                key={`${message.id}-citation-${index}`}
                                href={url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center justify-between rounded-lg bg-gray-900/70 px-2 py-1 text-gray-300 transition-colors hover:text-white"
                              >
                                <span>{`Source ${index + 1}`}</span>
                                <span className="truncate pl-3 text-[11px] text-gray-400">
                                  {getCitationLabel(url)}
                                </span>
                              </a>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}

                {chatStatus === "loading" && (
                  <div className="flex justify-start">
                    <div className="flex items-center space-x-2 rounded-xl border border-gray-700 bg-gray-800/70 px-3 py-2 text-sm text-gray-200">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Analyzing…</span>
                    </div>
                  </div>
                )}

                <div ref={messageEndRef} />
              </div>
            </div>

            <div className="border-t border-gray-800 bg-gray-900/30 px-5 py-4">
              {selectedText && (
                <div className="mb-3 rounded-xl border border-gray-700/70 bg-gray-800/60 px-3 py-3 text-sm text-gray-100 shadow-inner">
                  <div className="flex items-start justify-between gap-3">
                    <span className="text-xs font-semibold uppercase tracking-wide text-gray-400">
                      Selected text
                    </span>
                    <button
                      onClick={handleInsertSelectedText}
                      type="button"
                      className="rounded-lg bg-white px-3 py-1 text-xs font-semibold text-black shadow hover:bg-gray-200"
                    >
                      Add to chat
                    </button>
                  </div>
                  <p className="mt-2 max-h-24 overflow-y-auto text-sm leading-snug text-gray-200">
                    {selectedText}
                  </p>
                </div>
              )}
              <div className="flex space-x-2">
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Ask Relay anything..."
                  disabled={isSendDisabled}
                  className="flex-1 rounded-xl border border-gray-700 bg-gray-800/50 px-4 py-2.5 text-sm text-white placeholder-gray-500 transition focus:border-transparent focus:outline-none focus:ring-2 focus:ring-gray-500 disabled:cursor-not-allowed disabled:opacity-60"
                />
                <button
                  onClick={handleSend}
                  disabled={isSendDisabled}
                  className="flex h-11 w-11 items-center justify-center rounded-xl bg-white text-black shadow-lg transition hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-60"
                  type="button"
                >
                  {chatStatus === "loading" ? (
                    <Loader2 className="h-4 w-4 animate-spin text-black" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                  <span className="sr-only">Send message</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* <button
        onClick={() => setIsChatExpanded((prev) => !prev)}
        className="fixed bottom-6 right-4 z-40 rounded-full bg-white p-4 text-black shadow-2xl transition-transform hover:scale-110 hover:bg-gray-200"
        type="button"
        aria-expanded={isChatExpanded}
        aria-controls="relay-chat-panel"
      >
        {isChatExpanded ? (
          <ChevronRight className="h-5 w-5" />
        ) : (
          <Sparkles className="h-5 w-5" />
        )}
        <span className="sr-only">
          {isChatExpanded ? "Collapse chat" : "Expand chat"}
        </span>
      </button> */}
    </div>
  );
}
