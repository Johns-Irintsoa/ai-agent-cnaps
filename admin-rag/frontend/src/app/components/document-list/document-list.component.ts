import { Component, OnInit } from '@angular/core';
import { AdminDocument, DocumentPage, DocumentStatus } from '../../models/admin-document.model';
import { DocumentService } from '../../services/document.service';

type FilterTab = 'ALL' | DocumentStatus;

@Component({
  selector: 'app-document-list',
  templateUrl: './document-list.component.html',
  styleUrls: ['./document-list.component.scss']
})
export class DocumentListComponent implements OnInit {
  documents: AdminDocument[] = [];
  totalElements = 0;
  totalPages = 0;
  page = 0;
  pageSize = 20;

  activeTab: FilterTab = 'ALL';
  searchQuery = '';
  loading = true;
  error = false;

  tabs: { label: string; value: FilterTab }[] = [
    { label: 'Tous',      value: 'ALL' },
    { label: 'Traités',   value: 'SUCCESS' },
    { label: 'En cours',  value: 'PROCESSING' },
    { label: 'En attente',value: 'PENDING' },
    { label: 'Erreurs',   value: 'ERROR' }
  ];

  constructor(private documentService: DocumentService) {}

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading = true;
    this.error = false;
    const status = this.activeTab !== 'ALL' ? this.activeTab : undefined;
    this.documentService.getDocuments(this.page, this.pageSize, status, this.searchQuery)
      .subscribe({
        next: (result: DocumentPage) => {
          this.documents     = result.content;
          this.totalElements = result.page.totalElements;
          this.totalPages    = result.page.totalPages;
          this.loading       = false;
        },
        error: () => { this.error = true; this.loading = false; }
      });
  }

  selectTab(tab: FilterTab): void {
    this.activeTab = tab;
    this.page = 0;
    this.load();
  }

  onSearch(): void {
    this.page = 0;
    this.activeTab = 'ALL';
    this.load();
  }

  clearSearch(): void {
    this.searchQuery = '';
    this.load();
  }

  prevPage(): void {
    if (this.page > 0) { this.page--; this.load(); }
  }

  nextPage(): void {
    if (this.page < this.totalPages - 1) { this.page++; this.load(); }
  }

  delete(doc: AdminDocument): void {
    if (!confirm(`Supprimer "${doc.filename}" ?`)) return;
    this.documentService.delete(doc.id).subscribe({
      next: () => this.load(),
      error: () => alert('Erreur lors de la suppression.')
    });
  }

  retry(doc: AdminDocument): void {
    this.documentService.retry(doc.id).subscribe({
      next: () => this.load(),
      error: () => alert('Erreur lors du retry.')
    });
  }

  formatDate(iso: string): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric' });
  }

  formatSize(bytes: number): string {
    if (!bytes) return '—';
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} Ko`;
    return `${(bytes / 1024 / 1024).toFixed(1)} Mo`;
  }
}
