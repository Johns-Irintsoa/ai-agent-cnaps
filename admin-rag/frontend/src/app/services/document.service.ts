import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AdminDocument, DashboardStats, DocumentPage, DocumentStatus } from '../models/admin-document.model';

@Injectable({ providedIn: 'root' })
export class DocumentService {

  private readonly base = '/api/admin';

  constructor(private http: HttpClient) {}

  getStats(): Observable<DashboardStats> {
    return this.http.get<DashboardStats>(`${this.base}/dashboard/stats`);
  }

  getDocuments(page = 0, size = 20, status?: DocumentStatus, search?: string): Observable<DocumentPage> {
    let params = new HttpParams().set('page', page).set('size', size);
    if (status) params = params.set('status', status);
    if (search?.trim()) params = params.set('search', search.trim());
    return this.http.get<DocumentPage>(`${this.base}/documents`, { params });
  }

  getById(id: string): Observable<AdminDocument> {
    return this.http.get<AdminDocument>(`${this.base}/documents/${id}`);
  }

  uploadFile(file: File, uploadedBy = 'admin'): Observable<AdminDocument> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<AdminDocument>(`${this.base}/documents/upload`, form, {
      headers: { 'X-User-Id': uploadedBy }
    });
  }

  uploadUrl(sourceUrl: string, uploadedBy = 'admin'): Observable<AdminDocument> {
    const params = new HttpParams().set('source_url', sourceUrl);
    return this.http.post<AdminDocument>(`${this.base}/documents/upload`, null, {
      params,
      headers: { 'X-User-Id': uploadedBy }
    });
  }

  delete(id: string): Observable<void> {
    return this.http.delete<void>(`${this.base}/documents/${id}`);
  }

  retry(id: string): Observable<AdminDocument> {
    return this.http.post<AdminDocument>(`${this.base}/documents/${id}/retry`, {});
  }

  ingestWebPages(): Observable<{ status: string; message: string }> {
    return this.http.post<{ status: string; message: string }>(`${this.base}/documents/ingest-web-pages`, {});
  }
}
