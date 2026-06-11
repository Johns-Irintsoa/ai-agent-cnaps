export interface MessageMetadata {
  source_url?: string;
  title?: string;
  date_posted?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
  timestamp: Date;
  metadata?: MessageMetadata;
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
