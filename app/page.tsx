import Link from "next/link";
import { redirect } from "next/navigation";

import { auth } from "./(auth)/auth";

export default async function HomePage() {
  const session = await auth();

  if (session) {
    // If already signed in, take the user straight to the chat interface.
    redirect("/chat");
  }

  return (
    <div className="flex h-dvh w-screen items-center justify-center bg-background">
      <div className="mx-4 w-full max-w-2xl rounded-2xl bg-card/50 p-8 shadow-md">
        <div className="flex flex-col items-center gap-6 text-center">
          <h1 className="text-3xl font-semibold dark:text-zinc-50">Welcome to Hack Princeton</h1>
          <p className="max-w-xl text-gray-600 dark:text-zinc-400">
            Chat with your AI assistant. Sign in to continue or try a guest session.
          </p>

          <div className="flex gap-4">
            <Link
              href="/login"
              className="rounded-md bg-primary px-4 py-2 font-medium text-white hover:opacity-95"
            >
              Sign in
            </Link>

            <Link
              href="/register"
              className="rounded-md border border-gray-200 px-4 py-2 font-medium text-gray-800 dark:text-zinc-200 hover:bg-gray-100"
            >
              Sign up
            </Link>

            <Link
              href="/api/auth/guest"
              className="rounded-md bg-zinc-100 px-4 py-2 font-medium text-zinc-800 hover:opacity-95 dark:bg-zinc-800 dark:text-zinc-200"
            >
              Continue as guest
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
