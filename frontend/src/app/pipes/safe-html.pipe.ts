import { Pipe, PipeTransform } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

@Pipe({ name: 'safeHtml', standalone: true })
export class SafeHtmlPipe implements PipeTransform {
  constructor(private sanitizer: DomSanitizer) {}

  transform(text: string): SafeHtml {
    const escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    const linked = escaped.replace(
      /(https?:\/\/[^\s|<]+)/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer" class="inline-link">$1</a>'
    );

    const withBreaks = linked.replace(/\n/g, '<br>');

    return this.sanitizer.bypassSecurityTrustHtml(withBreaks);
  }
}
