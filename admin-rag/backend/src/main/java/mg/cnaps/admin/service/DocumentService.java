package mg.cnaps.admin.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import mg.cnaps.admin.dto.DashboardStats;
import mg.cnaps.admin.dto.DocumentResponse;
import mg.cnaps.admin.entity.*;
import mg.cnaps.admin.repository.DocumentLogRepository;
import mg.cnaps.admin.repository.DocumentRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
@RequiredArgsConstructor
@SuppressWarnings("null")
public class DocumentService {

    private final DocumentRepository documentRepository;
    private final DocumentLogRepository logRepository;
    private final FastApiClient fastApiClient;

    @Value("${rag.upload-dir}")
    private String uploadDir;

    private static final Pattern CHUNK_COUNT_PATTERN = Pattern.compile("^(\\d+)");

    // ─── Upload ───────────────────────────────────────────────────────────────

    @Transactional
    public DocumentResponse upload(MultipartFile file, String sourceUrl, String uploadedBy) throws IOException {
        String docId = UUID.randomUUID().toString();

        String filename;
        byte[] fileBytes = null;
        FileType fileType;
        long fileSize;
        String storedPath = null;

        if (file != null && !file.isEmpty()) {
            filename  = file.getOriginalFilename();
            fileBytes = file.getBytes();
            fileSize  = file.getSize();
            fileType  = detectFileType(filename);
            storedPath = saveFileToDisk(docId, filename, fileBytes);
        } else if (sourceUrl != null && !sourceUrl.isBlank()) {
            filename = sourceUrl;
            fileSize = 0;
            fileType = FileType.HTML;
        } else {
            throw new IllegalArgumentException("Fichier ou URL requis");
        }

        RagDocument doc = new RagDocument();
        doc.setId(docId);
        doc.setFilename(filename);
        doc.setFileType(fileType);
        doc.setFileSize(fileSize);
        doc.setFilePath(storedPath);
        doc.setSourceUrl(sourceUrl);
        doc.setStatus(DocumentStatus.PENDING);
        doc.setUploadedBy(uploadedBy);
        documentRepository.save(doc);

        logRepository.save(new RagDocumentLog(docId, "UPLOAD", "Document reçu : " + filename));

        // Appel FastAPI fire-and-forget : met à jour Oracle quand FastAPI répond
        if (fileType == FileType.HTML || (file == null || file.isEmpty())) {
            String url = sourceUrl != null ? sourceUrl : filename;
            fastApiClient.ingestUrl(url, List.of())
                    .subscribe(
                            response -> handleSuccess(docId, response),
                            error    -> markError(docId, "Echec ingestion URL : " + error.getMessage())
                    );
        } else {
            final byte[] bytesToSend = fileBytes;
            final String finalFilename = filename;
            fastApiClient.ingestPdf(finalFilename, bytesToSend)
                    .subscribe(
                            response -> handleSuccess(docId, response),
                            error    -> markError(docId, "Echec ingestion PDF : " + error.getMessage())
                    );
        }

        return DocumentResponse.from(doc);
    }

    // ─── Lecture (Oracle 11g : ROWNUM au lieu de FETCH FIRST) ────────────────

    public Page<DocumentResponse> findAll(Pageable pageable) {
        int[] rows = toRows(pageable);
        List<RagDocument> content = documentRepository.findAllNative(rows[0], rows[1]);
        long total = documentRepository.count();
        return new PageImpl<>(content, pageable, total).map(DocumentResponse::from);
    }

    public Page<DocumentResponse> findByStatus(DocumentStatus status, Pageable pageable) {
        int[] rows = toRows(pageable);
        List<RagDocument> content = documentRepository.findByStatusNative(status.name(), rows[0], rows[1]);
        long total = documentRepository.countByStatusNative(status.name());
        return new PageImpl<>(content, pageable, total).map(DocumentResponse::from);
    }

    public Page<DocumentResponse> searchByFilename(String filename, Pageable pageable) {
        int[] rows = toRows(pageable);
        List<RagDocument> content = documentRepository.findByFilenameNative(filename, rows[0], rows[1]);
        long total = documentRepository.countByFilenameNative(filename);
        return new PageImpl<>(content, pageable, total).map(DocumentResponse::from);
    }

    private int[] toRows(Pageable pageable) {
        int startRow = (int) pageable.getOffset();
        int endRow   = startRow + pageable.getPageSize();
        return new int[]{startRow, endRow};
    }

    public DocumentResponse findById(String id) {
        RagDocument doc = documentRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Document introuvable : " + id));
        List<DocumentResponse.LogEntry> logs = logRepository
                .findByDocumentIdOrderByCreatedAtAsc(id)
                .stream()
                .map(l -> new DocumentResponse.LogEntry(l.getStep(), l.getMessage(), l.getCreatedAt()))
                .toList();
        return DocumentResponse.from(doc, logs);
    }

    // ─── Dashboard ────────────────────────────────────────────────────────────

    public DashboardStats getDashboardStats() {
        long total      = documentRepository.count();
        long success    = documentRepository.countByStatus(DocumentStatus.SUCCESS);
        long error      = documentRepository.countByStatus(DocumentStatus.ERROR);
        long processing = documentRepository.countByStatus(DocumentStatus.PROCESSING);
        long pending    = documentRepository.countByStatus(DocumentStatus.PENDING);

        List<RagDocument> lastDocs = documentRepository.findTopDocument();

        String lastAt = "";
        String lastFilename = "";
        if (!lastDocs.isEmpty()) {
            RagDocument last = lastDocs.get(0);
            lastAt       = last.getUpdatedAt() != null
                    ? last.getUpdatedAt().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME) : "";
            lastFilename = last.getFilename();
        }

        return new DashboardStats(total, success, error, processing, pending, lastAt, lastFilename);
    }

    // ─── Suppression ─────────────────────────────────────────────────────────

    @Transactional
    public void delete(String id) {
        RagDocument doc = documentRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Document introuvable : " + id));

        if (doc.getFilePath() != null) {
            try { Files.deleteIfExists(Path.of(doc.getFilePath())); }
            catch (IOException e) {
                log.warn("Impossible de supprimer le fichier sur disque : {}", doc.getFilePath());
            }
        }

        String chromaSource = (doc.getFileType() == FileType.HTML && doc.getSourceUrl() != null)
                ? doc.getSourceUrl() : doc.getFilename();
        fastApiClient.deleteChunks(chromaSource).block();

        logRepository.deleteAll(logRepository.findByDocumentIdOrderByCreatedAtAsc(id));
        documentRepository.deleteById(id);
        log.info("Document {} supprimé d'Oracle et de ChromaDB", id);
    }

    // ─── Retry ────────────────────────────────────────────────────────────────

    @Transactional
    public DocumentResponse retry(String id) throws IOException {
        RagDocument doc = documentRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Document introuvable : " + id));
        if (doc.getStatus() != DocumentStatus.ERROR) {
            throw new IllegalStateException("Seuls les documents en erreur peuvent être relancés");
        }

        doc.setStatus(DocumentStatus.PENDING);
        doc.setErrorMessage(null);
        documentRepository.save(doc);
        logRepository.save(new RagDocumentLog(id, "RETRY", "Relance de l'indexation"));

        if (doc.getFileType() == FileType.HTML) {
            String url = doc.getSourceUrl() != null ? doc.getSourceUrl() : doc.getFilename();
            fastApiClient.ingestUrl(url, List.of())
                    .subscribe(
                            response -> handleSuccess(id, response),
                            error    -> markError(id, "Echec retry URL : " + error.getMessage())
                    );
        } else if (doc.getFilePath() != null) {
            byte[] fileBytes = Files.readAllBytes(Path.of(doc.getFilePath()));
            fastApiClient.ingestPdf(doc.getFilename(), fileBytes)
                    .subscribe(
                            response -> handleSuccess(id, response),
                            error    -> markError(id, "Echec retry PDF : " + error.getMessage())
                    );
        } else {
            markError(id, "Fichier introuvable pour le retry");
        }

        return DocumentResponse.from(doc);
    }

    // ─── Helpers privés ──────────────────────────────────────────────────────

    private void handleSuccess(String docId, Map<String, Object> response) {
        String message = response != null ? String.valueOf(response.getOrDefault("message", "")) : "";
        int chunkCount = parseChunkCount(message);
        documentRepository.findById(docId).ifPresent(doc -> {
            doc.setStatus(DocumentStatus.SUCCESS);
            doc.setChunkCount(chunkCount);
            doc.setProcessedAt(LocalDateTime.now());
            documentRepository.save(doc);
            logRepository.save(new RagDocumentLog(docId, "SUCCESS",
                    chunkCount > 0 ? chunkCount + " chunks indexés" : message));
            log.info("Indexation réussie pour doc_id={} ({} chunks)", docId, chunkCount);
        });
    }

    public void markError(String docId, String message) {
        documentRepository.findById(docId).ifPresent(doc -> {
            doc.setStatus(DocumentStatus.ERROR);
            doc.setErrorMessage(message);
            documentRepository.save(doc);
            logRepository.save(new RagDocumentLog(docId, "ERROR", message));
            log.error("Indexation échouée pour doc_id={} : {}", docId, message);
        });
    }

    private int parseChunkCount(String message) {
        if (message == null || message.isBlank()) return 0;
        Matcher m = CHUNK_COUNT_PATTERN.matcher(message.trim());
        return m.find() ? Integer.parseInt(m.group(1)) : 0;
    }

    private FileType detectFileType(String filename) {
        if (filename == null) return FileType.PDF;
        String lower = filename.toLowerCase();
        if (lower.endsWith(".pdf"))                        return FileType.PDF;
        if (lower.endsWith(".docx"))                       return FileType.DOCX;
        if (lower.endsWith(".xlsx"))                       return FileType.XLSX;
        if (lower.endsWith(".png"))                        return FileType.PNG;
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return FileType.JPG;
        if (lower.endsWith(".html") || lower.endsWith(".htm")) return FileType.HTML;
        return FileType.PDF;
    }

    private String saveFileToDisk(String docId, String filename, byte[] bytes) throws IOException {
        Path dir = Path.of(uploadDir, docId);
        Files.createDirectories(dir);
        Path dest = dir.resolve(filename);
        Files.write(dest, bytes);
        return dest.toString();
    }
}
