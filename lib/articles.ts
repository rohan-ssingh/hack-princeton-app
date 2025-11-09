import rawArticles from "@/app/articles.json";

export type ArticleRecord = {
  categoryName: string;
  title: string;
  summary: string;
  body: string;
  references: string[];
};

export type ArticleWithMeta = ArticleRecord & {
  slug: string;
};

const normalizeItem = (item: any): ArticleRecord => ({
  categoryName: String(item.category_name ?? ""),
  title: String(item.article_title ?? ""),
  summary: String(item.article_summary ?? ""),
  body: String(item.article_body ?? ""),
  references: Array.isArray(item.referenced_urls)
    ? item.referenced_urls.map((value) => String(value))
    : [],
});

const slugify = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

export const createArticleSlug = (category: string, title: string) => {
  return `${slugify(category)}-${slugify(title)}`;
};

const articles: ArticleRecord[] = Array.isArray(rawArticles)
  ? rawArticles.map(normalizeItem)
  : [];

export const getArticles = (): ArticleRecord[] => articles;

export const findArticleBySlug = (slug: string): ArticleWithMeta | null => {
  for (const article of articles) {
    const candidateSlug = createArticleSlug(article.categoryName, article.title);
    if (candidateSlug === slug) {
      return {
        ...article,
        slug: candidateSlug,
      };
    }
  }

  return null;
};
