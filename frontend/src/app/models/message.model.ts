export interface SourceItem {
  source_url: string;
  title?: string;
  date_posted?: string;
}

export interface MessageMetadata {
  source_url?: string;
  title?: string;
  date_posted?: string;
  sources?: SourceItem[];
}

export interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
  timestamp: Date;
  metadata?: MessageMetadata;
}

export interface ChatRequest {
  message: string;
  matricule?: string;
  password?: string;
}

export interface RAGResponse {
  answer: string;
  needs_auth: boolean;
  needs_matricule: boolean;
  metadata?: MessageMetadata;
  from_cache: boolean;
}

export interface SessionEntry {
  id: string;
  question: string;
  timestamp: Date;
}
