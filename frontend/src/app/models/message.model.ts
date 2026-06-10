export interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
  timestamp: Date;
}

export interface RAGResponse {
  answer: string;
  metadata?: {
    source_url?: string;
    title?: string;
    date_posted?: string;
  };
  from_cache: boolean;
}

export interface SessionEntry {
  id: string;
  question: string;
  timestamp: Date;
}
