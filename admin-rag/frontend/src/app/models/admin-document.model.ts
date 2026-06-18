export type DocumentStatus = 'PENDING' | 'PROCESSING' | 'SUCCESS' | 'ERROR';
export type FileType = 'PDF' | 'DOCX' | 'XLSX' | 'PNG' | 'JPG' | 'HTML';

export interface AdminDocument {
  id: string;
  filename: string;
  fileType: FileType;
  fileSize: number;
  sourceUrl?: string;
  status: DocumentStatus;
  errorMessage?: string;
  chunkCount: number;
  uploadedBy: string;
  importedAt: string;
  processedAt?: string;
  updatedAt?: string;
  logs?: DocumentLog[];
}

export interface DocumentLog {
  step: string;
  message: string;
  createdAt: string;
}

export interface DashboardStats {
  totalIndexed: number;
  totalSuccess: number;
  totalError: number;
  totalProcessing: number;
  totalPending: number;
  lastUpdatedAt: string;
  lastUpdatedFilename: string;
}

export interface DocumentPage {
  content: AdminDocument[];
  page: {
    size: number;
    number: number;
    totalElements: number;
    totalPages: number;
  };
}
