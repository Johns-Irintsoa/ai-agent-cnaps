package mg.cnaps.admin.controller;

import jakarta.validation.constraints.Min;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import mg.cnaps.admin.dto.DocumentResponse;
import mg.cnaps.admin.entity.DocumentStatus;
import mg.cnaps.admin.service.DocumentService;
import mg.cnaps.admin.service.FastApiClient;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/admin/documents")
@RequiredArgsConstructor
public class DocumentController {

    private final DocumentService documentService;
    private final FastApiClient fastApiClient;

    @PostMapping("/ingest-web-pages")
    public ResponseEntity<Map<String, String>> ingestWebPages() {
        try {
            fastApiClient.ingestWebPages().subscribe(
                r  -> log.info("Pages CNaPS indexees : {}", r),
                e  -> log.error("Echec indexation pages CNaPS : {}", e.getMessage())
            );
            return ResponseEntity.accepted()
                    .body(Map.of("status", "accepted",
                                 "message", "Indexation des pages CNaPS lancee en arriere-plan."));
        } catch (Exception e) {
            log.error("Erreur ingestWebPages", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    @PostMapping("/upload")
    public ResponseEntity<DocumentResponse> upload(
            @RequestPart(value = "file", required = false) MultipartFile file,
            @RequestParam(value = "source_url", required = false) String sourceUrl,
            @RequestHeader(value = "X-User-Id", defaultValue = "admin") String userId
    ) {
        try {
            DocumentResponse response = documentService.upload(file, sourceUrl, userId);
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (IllegalArgumentException e) {
            log.warn("Upload invalide : {}", e.getMessage());
            return ResponseEntity.badRequest().build();
        } catch (Exception e) {
            log.error("Erreur lors de l'upload", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    @GetMapping
    public ResponseEntity<Page<DocumentResponse>> list(
            @RequestParam(defaultValue = "0") @Min(0) int page,
            @RequestParam(defaultValue = "20") @Min(1) int size,
            @RequestParam(required = false) DocumentStatus status,
            @RequestParam(required = false) String search
    ) {
        try {
            PageRequest pageable = PageRequest.of(page, size, Sort.by("importedAt").descending());
            Page<DocumentResponse> result;
            if (search != null && !search.isBlank()) {
                result = documentService.searchByFilename(search, pageable);
            } else if (status != null) {
                result = documentService.findByStatus(status, pageable);
            } else {
                result = documentService.findAll(pageable);
            }
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            log.error("Erreur lors de la liste des documents", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    @GetMapping("/{id}")
    public ResponseEntity<DocumentResponse> getById(@PathVariable String id) {
        try {
            return ResponseEntity.ok(documentService.findById(id));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            log.error("Erreur lors de la récupération du document {}", id, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable String id) {
        try {
            documentService.delete(id);
            return ResponseEntity.noContent().build();
        } catch (IllegalArgumentException e) {
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            log.error("Erreur lors de la suppression du document {}", id, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    @PostMapping("/{id}/retry")
    public ResponseEntity<DocumentResponse> retry(@PathVariable String id) {
        try {
            return ResponseEntity.ok(documentService.retry(id));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.notFound().build();
        } catch (IllegalStateException e) {
            return ResponseEntity.status(HttpStatus.CONFLICT).build();
        } catch (Exception e) {
            log.error("Erreur lors du retry du document {}", id, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}
