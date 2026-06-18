import { Component, OnInit } from '@angular/core';
import { DashboardStats } from '../../models/admin-document.model';
import { DocumentService } from '../../services/document.service';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit {
  stats: DashboardStats | null = null;
  loading = true;
  error = false;

  constructor(private documentService: DocumentService) {}

  ngOnInit(): void {
    this.loadStats();
  }

  loadStats(): void {
    this.loading = true;
    this.error = false;
    this.documentService.getStats().subscribe({
      next: (data) => { this.stats = data; this.loading = false; },
      error: ()     => { this.error = true;  this.loading = false; }
    });
  }

  formatDate(iso: string): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('fr-FR');
  }
}
