import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { interval, Subscription } from 'rxjs';
import { switchMap, takeWhile } from 'rxjs/operators';
import { AdminDocument } from '../../models/admin-document.model';
import { DocumentService } from '../../services/document.service';

type UploadStep = 'idle' | 'uploading' | 'pending' | 'processing' | 'done' | 'error';

@Component({
  selector: 'app-document-upload',
  templateUrl: './document-upload.component.html',
  styleUrls: ['./document-upload.component.scss']
})
export class DocumentUploadComponent {
  mode: 'file' | 'url' | 'web' = 'file';
  webState: 'idle' | 'loading' | 'done' | 'error' = 'idle';
  webMessage = '';
  selectedFile: File | null = null;
  sourceUrl = '';
  step: UploadStep = 'idle';
  errorMessage = '';
  uploadedDoc: AdminDocument | null = null;
  isDragging = false;

  private pollSub?: Subscription;

  constructor(private documentService: DocumentService, private router: Router) {}

  onDragOver(e: DragEvent): void {
    e.preventDefault();
    this.isDragging = true;
  }

  onDragLeave(): void {
    this.isDragging = false;
  }

  onDrop(e: DragEvent): void {
    e.preventDefault();
    this.isDragging = false;
    const file = e.dataTransfer?.files[0];
    if (file) this.setFile(file);
  }

  onFileSelect(e: Event): void {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) this.setFile(file);
  }

  private setFile(file: File): void {
    this.selectedFile = file;
    this.step = 'idle';
    this.errorMessage = '';
  }

  submit(): void {
    this.step = 'uploading';
    this.errorMessage = '';

    const upload$ = this.mode === 'file' && this.selectedFile
      ? this.documentService.uploadFile(this.selectedFile)
      : this.documentService.uploadUrl(this.sourceUrl);

    upload$.subscribe({
      next: (doc) => {
        this.uploadedDoc = doc;
        this.step = 'pending';
        this.startPolling(doc.id);
      },
      error: (err) => {
        this.step = 'error';
        this.errorMessage = err?.error?.detail ?? 'Erreur lors de l\'envoi.';
      }
    });
  }

  private startPolling(docId: string): void {
    this.pollSub = interval(3000).pipe(
      switchMap(() => this.documentService.getById(docId)),
      takeWhile((doc) => doc.status === 'PENDING' || doc.status === 'PROCESSING', true)
    ).subscribe({
      next: (doc) => {
        this.uploadedDoc = doc;
        if (doc.status === 'PROCESSING') this.step = 'processing';
        if (doc.status === 'SUCCESS')    this.step = 'done';
        if (doc.status === 'ERROR') {
          this.step = 'error';
          this.errorMessage = doc.errorMessage ?? 'Echec de l\'indexation.';
        }
      }
    });
  }

  reset(): void {
    this.pollSub?.unsubscribe();
    this.step = 'idle';
    this.selectedFile = null;
    this.sourceUrl = '';
    this.uploadedDoc = null;
    this.errorMessage = '';
  }

  goToDocuments(): void {
    this.router.navigate(['/documents']);
  }

  submitWebPages(): void {
    this.webState = 'loading';
    this.documentService.ingestWebPages().subscribe({
      next: (res) => { this.webState = 'done'; this.webMessage = res.message; },
      error: ()    => { this.webState = 'error'; this.webMessage = 'Erreur lors du lancement.'; }
    });
  }

  get canSubmit(): boolean {
    if (this.step !== 'idle') return false;
    return this.mode === 'file' ? !!this.selectedFile : this.sourceUrl.startsWith('http');
  }

  formatSize(bytes: number): string {
    if (!bytes) return '—';
    if (bytes < 1024) return `${bytes} o`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`;
    return `${(bytes / 1024 / 1024).toFixed(1)} Mo`;
  }
}
