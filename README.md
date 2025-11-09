<h1 align="center">Relay</h1>

<p align="center">
  <b>Your personal analyst for local and state politics.</b>
</p>

<p align="center">
  Making local government radically transparent. Relay is a scalable infrastructure for democracy, designed to keep you informed about the legislative activities that directly impact your life.
</p>

<p align="center">
  <a href="#the-problem"><strong>The Problem</strong></a> ·
  <a href="#our-solution"><strong>Our Solution</strong></a> ·
  <a href="#features"><strong>Features</strong></a> ·
  <a href="#system-architecture"><strong>System Architecture</strong></a>
</p>
<br/>

## The Problem

What affects citizens most directly is not the national news, but the legislative activity within their own state and local governments. However, with the decline of local news outlets, 70 million Americans are left without a dedicated source for this critical information. This information vacuum leads to a reliance on often-biased national narratives, increasing party polarization and leaving citizens disengaged from the decisions that shape their communities.

## Our Solution

Relay bridges this gap by providing a scalable infrastructure for democracy. We deliver personalized, bias-free alerts and summaries about legislative activity at the state and local levels. Our platform analyzes primary sources like official transcripts of government meetings to provide fact-based information tailored to the issues you care about.

This approach helps reduce party polarization, democratizes civic intelligence, and ensures that legislation across the country is more transparent and sustainable. The system can be applied to analyze school board meetings, local council meetings, and other government proceedings, even in remote areas.

## Features

- **Personalized Alerts & Newsletters**: Receive email alerts and monthly newsletters about legislative updates that match your specific interests and location.
- **Bias-Free Content**: Articles are generated directly from the primary sources of legislature (e.g., transcripts) to eliminate bias and help you form your own opinions.
- **Customized Reading Level**: Adjust the complexity of the content to your preference, making civic engagement accessible to everyone.
- **Interactive RAG Chatbot**: Ask questions and get answers with direct citations from legislative documents. The chatbot understands your priorities and reading level for a truly personalized experience.
- **Personalized Feed**: An in-app feed that surfaces snippets of information most important to you, with options to filter by state, local, or school board levels.

## System Architecture

Relay is built on a robust system of data processing, AI agents, and a powerful database to deliver timely and relevant information.

### Database and Data Ingestion
- **Data Source**: We start by scraping video transcripts and metadata from government proceedings (e.g., Vermont State Legislature).
- **Database**: Transcripts and their metadata (committee, date, etc.) are stored in a structured database.
- **Vector Embeddings**: We use vector embeddings on the transcripts to enable efficient semantic search and retrieval based on user interests.

### AI Agent Swarm (using Dedalus MCP)
A swarm of specialized AI agents works together to power the platform:
- **User Preferences Agent**: Captures and interprets a user's interests, political ideology, location, and reading level from an initial unstructured description.
- **Alert Determination Agent**: Scans new transcripts to determine if they contain information relevant to a user's priorities.
- **Extraction Agent**: Pulls all relevant details from a transcript based on the user's interests.
- **Article Agent**: Generates a full, easy-to-read article from the extracted information.
- **Reading Level Agent**: Adjusts the generated content to the user's specified reading level.
- **Monthly Newsletter Agent**: Aggregates all relevant articles and updates from the past month into a personalized newsletter.
- **Question Answering Bot**: A RAG-based chatbot that answers user queries with citations from the knowledge base.

### User Experience
- **Onboarding**: A new user describes their interests, which our AI distills into a set of priorities.
- **Out-of-App**: Users receive "Alert" emails for urgent, relevant updates and a comprehensive "Newsletter" email at the end of each month.
- **In-App**: A personalized feed provides the most recent and relevant legislative updates. A chatbot is available to answer questions and provide deeper insights.

