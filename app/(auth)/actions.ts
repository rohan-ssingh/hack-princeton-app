"use server";

import { z } from "zod";

import { createUser, getUser } from "@/lib/db/queries";

import { signIn } from "./auth";

const authFormSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
});

export type LoginActionState = {
  status: "idle" | "in_progress" | "success" | "failed" | "invalid_data";
};

export const login = async (
  _: LoginActionState,
  formData: FormData
): Promise<LoginActionState> => {
  try {
    const validatedData = authFormSchema.parse({
      email: formData.get("email"),
      password: formData.get("password"),
    });

    await signIn("credentials", {
      email: validatedData.email,
      password: validatedData.password,
      redirect: false,
    });

    return { status: "success" };
  } catch (error) {
    if (error instanceof z.ZodError) {
      return { status: "invalid_data" };
    }

    return { status: "failed" };
  }
};

export type RegisterActionState = {
  status:
    | "idle"
    | "in_progress"
    | "success"
    | "failed"
    | "user_exists"
    | "invalid_data";
};

export const register = async (
  _: RegisterActionState,
  formData: FormData
): Promise<RegisterActionState> => {
  try {
    // Email and password validation and database logic - UNCHANGED
    const validatedData = authFormSchema.parse({
      email: formData.get("email"),
      password: formData.get("password"),
    });

    const [existingUser] = await getUser(validatedData.email);

    if (existingUser) {
      return { status: "user_exists" } as RegisterActionState;
    }

    // Gather profile fields before persisting to the database
    const selectedTopics = formData.getAll("topics") as string[];
    const otherTopics = formData.get("otherTopics") as string | null;
    const readingLevel = formData.get("readingLevel") as string | null;
    const locations = formData.get("locations") as string | null;

    const isOtherSelected = selectedTopics.includes("other");
    const combinedTopics = isOtherSelected && otherTopics
      ? [...selectedTopics.filter((t) => t !== "other"), otherTopics.trim()]
      : selectedTopics;
    const normalizedTopics = combinedTopics
      .map((topic) => topic.trim())
      .filter((topic) => topic.length > 0);
    const normalizedLocations = locations?.trim() ?? null;
    const depth = readingLevel ? Number.parseInt(readingLevel, 10) : null;

    await createUser({
      email: validatedData.email,
      password: validatedData.password,
      locations: normalizedLocations,
      topics: normalizedTopics,
      depth: Number.isNaN(depth) ? null : depth,
    });

    // Sign in logic - UNCHANGED
    await signIn("credentials", {
      email: validatedData.email,
      password: validatedData.password,
      redirect: false,
    });

    return { status: "success" };
  } catch (error) {
    if (error instanceof z.ZodError) {
      return { status: "invalid_data" };
    }

    return { status: "failed" };
  }
};
