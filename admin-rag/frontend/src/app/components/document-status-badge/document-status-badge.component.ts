import { Component, Input } from '@angular/core';
import { DocumentStatus } from '../../models/admin-document.model';

@Component({
  selector: 'app-document-status-badge',
  templateUrl: './document-status-badge.component.html',
  styleUrls: ['./document-status-badge.component.scss']
})
export class DocumentStatusBadgeComponent {
  @Input() status!: DocumentStatus;

  get label(): string {
    const labels: Record<DocumentStatus, string> = {
      PENDING:    'En attente',
      PROCESSING: 'En cours',
      SUCCESS:    'Traité',
      ERROR:      'Erreur'
    };
    return labels[this.status] ?? this.status;
  }

  get cssClass(): string {
    const classes: Record<DocumentStatus, string> = {
      PENDING:    'badge--pending',
      PROCESSING: 'badge--processing',
      SUCCESS:    'badge--success',
      ERROR:      'badge--error'
    };
    return classes[this.status] ?? '';
  }
}
