package mg.cnaps.admin.dto;

import mg.cnaps.admin.entity.DocumentStatus;
import mg.cnaps.admin.entity.FileType;
import mg.cnaps.admin.entity.RagDocument;

import java.time.LocalDateTime;
import java.util.List;

public record DocumentResponse(
        String id,
        String filename,
        FileType fileType,
        long fileSize,
        String sourceUrl,
        DocumentStatus status,
        String errorMessage,
        int chunkCount,
        String uploadedBy,
        LocalDateTime importedAt,
        LocalDateTime processedAt,
        LocalDateTime updatedAt,
        List<LogEntry> logs
) {
    public record LogEntry(String step, String message, LocalDateTime createdAt) {}

    public static DocumentResponse from(RagDocument doc) {
        return new DocumentResponse(
                doc.getId(),
                doc.getFilename(),
                doc.getFileType(),
                doc.getFileSize(),
                doc.getSourceUrl(),
                doc.getStatus(),
                doc.getErrorMessage(),
                doc.getChunkCount(),
                doc.getUploadedBy(),
                doc.getImportedAt(),
                doc.getProcessedAt(),
                doc.getUpdatedAt(),
                List.of()
        );
    }

    public static DocumentResponse from(RagDocument doc, List<LogEntry> logs) {
        return new DocumentResponse(
                doc.getId(),
                doc.getFilename(),
                doc.getFileType(),
                doc.getFileSize(),
                doc.getSourceUrl(),
                doc.getStatus(),
                doc.getErrorMessage(),
                doc.getChunkCount(),
                doc.getUploadedBy(),
                doc.getImportedAt(),
                doc.getProcessedAt(),
                doc.getUpdatedAt(),
                logs
        );
    }
}
