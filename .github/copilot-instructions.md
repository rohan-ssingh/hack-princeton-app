## Repo snapshot for AI coding assistants

This is a Next.js (App Router) AI Chatbot template extended in this repo. The goal of these notes is to make an AI coding agent productive quickly by pointing to architecture, conventions, and developer workflows unique to this project.

Key points (short):
- Framework: Next.js App Router (canary v15 in package.json). Code uses React Server Components (RSC) and Server Actions.
- AI integration: Uses the AI SDK and Vercel AI Gateway by default. See `lib/ai/*` (providers, prompts) and `app/(chat)/actions.ts` for examples.
- DB: Drizzle ORM; migrations are run with `lib/db/migrate.ts` and scripts in `package.json` (e.g. `pnpm db:migrate`).
- Package manager & scripts: `pnpm` is used. Primary commands: `pnpm dev`, `pnpm build` (runs migrations then `next build`), `pnpm test` (Playwright).
- Lint/format: uses `ultracite` via `pnpm lint`/`pnpm format`.

Important files & locations to reference:
- App entry & layout: `app/layout.tsx` — shows SessionProvider, ThemeProvider and global scripts.
- Auth actions: `app/(auth)/actions.ts` — server actions for register/login (example of Server Action patterns).
- Chat actions: `app/(chat)/actions.ts` — uses `ai.generateText` and demonstrates cookie + DB interactions for chat flows.
- AI plumbing: `lib/ai/providers.ts`, `lib/ai/prompts.ts` (look here to add or switch model providers or prompts).
- DB layer: `lib/db/*` and `lib/db/migrate.ts` — Drizzle migrations and queries live under `lib/db` and `lib/db/queries.ts`.
- Shared utilities/types: `lib/utils.ts`, `lib/constants.ts`, `lib/types.ts`.
- UI: `components/` contains the majority of client UI components (chat UI, editors, preview, etc.) and follows small, focused components.

Developer workflows & commands (explicit):
- Install: `pnpm install`
- Run dev: `pnpm dev` (dev server uses `next dev --turbo`)
- Build (local): `pnpm build` — note this runs `tsx lib/db/migrate` before `next build`.
- Start production server locally: `pnpm start`
- Database migration: `pnpm db:migrate` (runs `npx tsx lib/db/migrate.ts`). Use other drizzle-kit scripts from `package.json` as needed (e.g. `db:generate`, `db:studio`).
- Tests: `pnpm test` (sets `PLAYWRIGHT=True` then runs Playwright tests). Tests expect Playwright environment variables when running in CI.

Project-specific conventions & patterns
- Server Actions: server-side logic uses `"use server"` in `app/.../actions.ts`. These return small state results (see `LoginActionState`/`RegisterActionState`). When modifying auth or chat flows, follow the same return status enum pattern.
- AI usage: use the reusable `myProvider` (see `lib/ai/providers`) and the `generateText` flow from `app/(chat)/actions.ts`. Prompts are centralized in `lib/ai/prompts.ts`.
- DB access: queries are wrapped under `lib/db/queries.ts` and consumed by server actions. Prefer query helpers over raw SQL in server actions.
- Client vs Server: prefer keeping heavy AI / DB code on the server side (server actions or API routes). Client components and hooks live in `components/` and `hooks/` respectively.
- Error handling: utilities in `lib/errors.ts` and `lib/utils.ts` provide project-specific error types and wrappers (e.g., `ChatSDKError`). Reuse them when throwing or interpreting errors.
- Styling: Tailwind + shadcn pattern; component classes often use `cn()` from `lib/utils.ts` and `tailwind-merge`.

Integration & external dependencies
- Vercel AI Gateway / AI SDK: configured via `lib/ai` and environment variables. For non-Vercel deployments, set `AI_GATEWAY_API_KEY` or swap to a direct provider.
- Storage: `@vercel/blob` is used for file storage integrations.
- Auth: `next-auth` (see `app/(auth)` and `auth.ts` in that folder).
- DB: Postgres (Neon) + Drizzle. Look at `drizzle.config.ts` in the repo root for connection configuration.

Notes for an AI editing agent (what to do & avoid)
- Do: make changes inside `lib/` or `app/` with small, focused edits. When adding/altering server actions, update both the server action and any client component that consumes its state.
- Do: run the relevant scripts after changes: `pnpm dev`, `pnpm db:migrate`, and unit/e2e `pnpm test` (Playwright) if you touch interactive flows.
- Avoid: moving AI or DB logic from server to client. This project keeps sensitive keys and heavy compute on the server.
- Prefer conservative refactors. Many components and hooks are small and re-used; update usages when changing shapes (types) of returned data.

Quick examples (where to look)
- Add a new Server Action: copy pattern in `app/(auth)/actions.ts` and return a small union-status type (e.g. `status: 'idle'|'in_progress'|'success'|'failed'`).
- Add an AI prompt or provider: edit `lib/ai/prompts.ts` and `lib/ai/providers.ts`. Use `app/(chat)/actions.ts` as a usage example for `generateText`.
- Run a DB migration before build: `pnpm db:migrate` or `pnpm build` (build already runs migrations).

Files that exemplify important decisions
- `app/layout.tsx` — global providers and hydration notes.
- `app/(auth)/actions.ts`, `app/(chat)/actions.ts` — server actions patterns.
- `lib/ai/*`, `lib/db/*`, `lib/utils.ts` — where providers, queries and shared helpers live.
- `package.json` — scripts you must run (`dev`, `build`, `db:migrate`, `test`, `lint`).

If anything above is unclear or you'd like a different level of detail (more file-level examples or a short checklist for adding a new feature), tell me which area to expand and I will iterate.
