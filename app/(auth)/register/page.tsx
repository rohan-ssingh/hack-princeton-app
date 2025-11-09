"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { useActionState, useEffect, useState, type ChangeEvent } from "react";
import Form from "next/form";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SubmitButton } from "@/components/submit-button";
import { toast } from "@/components/toast";
import { type RegisterActionState, register } from "../actions";

const TOPICS = [
  "Housing & Development",
  "Education Funding & Property Tax",
  "Taxes & Economic Policy",
  "Environment & Climate",
  "Workforce & Labor",
  "Healthcare & Mental Health",
  "Public Safety & Justice",
  "Infrastructure & Energy",
  "Civic & Electoral Reform",
];

const READING_LEVELS = [
  {
    value: "1",
    label: "Level 1: Clear & concise",
    description: "Simple, easy-to-understand explanations",
  },
  {
    value: "2",
    label: "Level 2: Detailed & technical",
    description: "In-depth analysis with technical details",
  },
  {
    value: "3",
    label: "Level 3: Highly technical, analytical, or policy-style",
    description: "Expert-level analysis and policy documentation",
  },
];

export default function Page() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [isSuccessful, setIsSuccessful] = useState(false);
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [otherTopics, setOtherTopics] = useState("");
  const [selectedReadingLevel, setSelectedReadingLevel] = useState<string>("");

  const [state, formAction] = useActionState<RegisterActionState, FormData>(
    register,
    {
      status: "idle",
    }
  );

  const { update: updateSession } = useSession();

  useEffect(() => {
    if (state.status === "user_exists") {
      toast({ type: "error", description: "Account created, please login!" });
    } else if (state.status === "failed") {
      toast({ type: "error", description: "Account created, please login!" });
    } else if (state.status === "invalid_data") {
      toast({
        type: "error",
        description: "Failed validating your submission",
      });
    } else if (state.status === "success") {
      toast({ type: "success", description: "Account created successfully!" });

      setIsSuccessful(true);
      updateSession();
      router.refresh();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.status]);

  const handleTopicChange = (topic: string) => {
    if (topic === "other") {
      if (selectedTopics.includes("other")) {
        setSelectedTopics(selectedTopics.filter((t: string) => t !== "other"));
        setOtherTopics("");
      } else {
        setSelectedTopics([...selectedTopics, "other"]);
      }
    } else {
      if (selectedTopics.includes(topic)) {
        setSelectedTopics(selectedTopics.filter((t: string) => t !== topic));
      } else {
        setSelectedTopics([...selectedTopics, topic]);
      }
    }
  };

  const handleSubmit = (formData: FormData) => {
    setEmail(formData.get("email") as string);

    if (selectedTopics.includes("other") && otherTopics) {
      formData.append("otherTopics", otherTopics);
    }

    formAction(formData);
  };

  return (
    <div className="flex h-dvh w-screen items-start justify-center bg-background pt-8 md:items-center md:pt-0">
      <div className="flex w-full max-w-4xl flex-col gap-4 overflow-y-auto rounded-2xl px-4 pb-8 sm:px-16">
        <div className="flex flex-col items-center justify-center gap-1 px-4 text-center sm:px-16">
          <h3 className="font-semibold text-xl dark:text-zinc-50">Sign Up</h3>
          <p className="text-gray-500 text-sm dark:text-zinc-400">
            Create an account with your email and password
          </p>
        </div>

        <Form action={handleSubmit} className="flex flex-col gap-4 px-4 sm:px-16">
          {/* Email and Password - Keep existing implementation */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="flex flex-col gap-2">
              <Label
                className="font-normal text-zinc-600 dark:text-zinc-400"
                htmlFor="email"
              >
                Email Address
              </Label>
              <Input
                autoComplete="email"
                autoFocus
                className="bg-muted text-md md:text-sm"
                defaultValue={email}
                id="email"
                name="email"
                placeholder="user@acme.com"
                required
                type="email"
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label
                className="font-normal text-zinc-600 dark:text-zinc-400"
                htmlFor="password"
              >
                Password
              </Label>
              <Input
                className="bg-muted text-md md:text-sm"
                id="password"
                name="password"
                required
                type="password"
              />
            </div>
          </div>

          {/* First Name and Last Name */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="flex flex-col gap-2">
              <Label
                className="font-normal text-zinc-600 dark:text-zinc-400"
                htmlFor="firstName"
              >
                First Name
              </Label>
              <Input
                className="bg-muted text-md md:text-sm"
                id="firstName"
                name="firstName"
                placeholder="John"
                required
                type="text"
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label
                className="font-normal text-zinc-600 dark:text-zinc-400"
                htmlFor="lastName"
              >
                Last Name
              </Label>
              <Input
                className="bg-muted text-md md:text-sm"
                id="lastName"
                name="lastName"
                placeholder="Doe"
                required
                type="text"
              />
            </div>
          </div>

          {/* Topics Selection */}
          <div className="flex flex-col gap-2">
            <Label className="font-normal text-zinc-600 dark:text-zinc-400">
              Select Topics of Interest (Select all that apply)
            </Label>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {TOPICS.map((topic) => (
                <label
                  key={topic}
                  className="flex items-center gap-2 rounded-md border border-input bg-background p-2 hover:bg-muted cursor-pointer"
                >
                  <input
                    checked={selectedTopics.includes(topic)}
                    className="size-4 rounded border-gray-300 text-primary focus:ring-2 focus:ring-primary"
                    name="topics"
                    onChange={() => handleTopicChange(topic)}
                    type="checkbox"
                    value={topic}
                  />
                  <span className="text-sm text-foreground">{topic}</span>
                </label>
              ))}
              <label className="flex items-center gap-2 rounded-md border border-input bg-background p-2 hover:bg-muted cursor-pointer">
                <input
                  checked={selectedTopics.includes("other")}
                  className="size-4 rounded border-gray-300 text-primary focus:ring-2 focus:ring-primary"
                  name="topics"
                  onChange={() => handleTopicChange("other")}
                  type="checkbox"
                  value="other"
                />
                <span className="text-sm text-foreground">Other</span>
              </label>
            </div>

            {/* Other Topics Input */}
            {selectedTopics.includes("other") && (
              <div className="flex flex-col gap-2">
                <Label
                  className="font-normal text-zinc-600 dark:text-zinc-400"
                  htmlFor="otherTopics"
                >
                  Please specify your topics
                </Label>
                <Input
                  className="bg-muted text-md md:text-sm"
                  id="otherTopics"
                  name="otherTopics"
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    setOtherTopics(e.target.value)
                  }
                  placeholder="Enter your topics here"
                  type="text"
                  value={otherTopics}
                />
              </div>
            )}
          </div>

          {/* Reading Level and Locations - Side by side on larger screens */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {/* Reading Level Selection */}
            <div className="flex flex-col gap-2">
              <Label className="font-normal text-zinc-600 dark:text-zinc-400">
                How in-depth would you like your writing to be?
              </Label>
              <div className="flex flex-col gap-2">
                {READING_LEVELS.map((level) => (
                  <label
                    key={level.value}
                    className={`flex items-start gap-2 rounded-md border p-2 hover:bg-muted cursor-pointer ${
                      selectedReadingLevel === level.value
                        ? "border-primary bg-primary/5"
                        : "border-input bg-background"
                    }`}
                  >
                    <input
                      checked={selectedReadingLevel === level.value}
                      className="mt-1 size-4 border-gray-300 text-primary focus:ring-2 focus:ring-primary"
                      name="readingLevel"
                      onChange={() => setSelectedReadingLevel(level.value)}
                      required
                      type="radio"
                      value={level.value}
                    />
                    <div className="flex flex-col gap-0.5">
                      <span className="text-sm font-medium text-foreground">
                        {level.label}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {level.description}
                      </span>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Locations Input */}
            <div className="flex flex-col gap-2">
              <Label
                className="font-normal text-zinc-600 dark:text-zinc-400"
                htmlFor="locations"
              >
                What cities or states would you like insights on?
              </Label>
              <Input
                className="bg-muted text-md md:text-sm"
                id="locations"
                name="locations"
                placeholder="San Francisco, CA"
                required
                type="text"
              />
              <p className="text-xs text-muted-foreground">
                Please enter in the format: city, state
              </p>
            </div>
          </div>

          <SubmitButton isSuccessful={isSuccessful}>
            Sign Up
          </SubmitButton>

          <p className="mt-4 text-center text-gray-600 text-sm dark:text-zinc-400">
            {"Already have an account? "}
            <Link
              className="font-semibold text-gray-800 hover:underline dark:text-zinc-200"
              href="/login"
            >
              Sign in
            </Link>
            {" instead."}
          </p>
        </Form>
      </div>
    </div>
  );
}
