package mg.cnaps.admin.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import mg.cnaps.admin.dto.DashboardStats;
import mg.cnaps.admin.service.DocumentService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/api/admin/dashboard")
@RequiredArgsConstructor
public class DashboardController {

    private final DocumentService documentService;

    @GetMapping("/stats")
    public ResponseEntity<DashboardStats> getStats() {
        try {
            return ResponseEntity.ok(documentService.getDashboardStats());
        } catch (Exception e) {
            log.error("Erreur lors du calcul des stats dashboard", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}
