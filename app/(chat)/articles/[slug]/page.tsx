import Link from "next/link";
import { notFound } from "next/navigation";
import { findArticleBySlug } from "@/lib/articles";

const renderParagraphs = (content: string) => {
  return content
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
};

export default async function Page(props: { params: Promise<{ slug: string }> }) {
  const params = await props.params;
  const { slug } = params;

  const article = findArticleBySlug(slug);

  if (!article) {
    notFound();
  }

  const paragraphs = renderParagraphs(article.body);

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-950 to-black text-gray-100">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-6 py-12">
        <Link
          className="text-sm font-semibold text-gray-400 transition hover:text-gray-100"
          href="/"
        >
          ← Back to feed
        </Link>

        <header className="space-y-4">
          <span className="inline-flex rounded-full border border-gray-800 bg-gray-900/80 px-4 py-1 text-xs uppercase tracking-wide text-gray-300">
            {article.categoryName}
          </span>
          <h1 className="text-3xl font-bold leading-tight text-white">
            {article.title}
          </h1>
          <p className="text-gray-300">{article.summary}</p>
        </header>

        <article className="prose prose-invert max-w-none space-y-4 text-gray-200">
          {paragraphs.map((paragraph, index) => (
            <p key={`paragraph-${index}`}>{paragraph}</p>
          ))}
        </article>

        {article.references.length > 0 && (
          <section className="space-y-3 border-t border-gray-800 pt-6">
            <h2 className="text-lg font-semibold text-white">Sources</h2>
            <ul className="space-y-2 text-sm text-gray-300">
              {article.references.map((reference) => (
                <li key={reference}>
                  <a
                    className="underline-offset-4 transition hover:text-white hover:underline"
                    href={reference}
                    rel="noopener noreferrer"
                    target="_blank"
                  >
                    {reference}
                  </a>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}
