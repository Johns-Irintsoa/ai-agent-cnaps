package mg.cnaps.admin.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;
import java.util.Objects;

@Slf4j
@Service
@RequiredArgsConstructor
@SuppressWarnings("null")
public class FastApiClient {

    private static final ParameterizedTypeReference<Map<String, Object>> MAP_TYPE =
            new ParameterizedTypeReference<>() {};

    private final WebClient fastApiWebClient;

    /**
     * Appelle POST /ingest-pdf avec le fichier en multipart.
     * FastAPI traite de façon synchrone et répond quand l'indexation est terminée.
     * Réponse attendue : {"status": "success", "message": "..."}
     */
    public Mono<Map<String, Object>> ingestPdf(String filename, byte[] fileBytes) {
        Objects.requireNonNull(fileBytes, "fileBytes ne peut pas être null");

        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("file", new ByteArrayResource(fileBytes) {
            @Override
            public String getFilename() { return filename; }
        }).header(HttpHeaders.CONTENT_DISPOSITION,
                "form-data; name=\"file\"; filename=\"" + filename + "\"");

        return fastApiWebClient.post()
                .uri("/ingest-pdf")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(builder.build()))
                .retrieve()
                .bodyToMono(MAP_TYPE);
    }

    /**
     * Appelle POST /ingestion/url avec un corps JSON.
     * FastAPI traite de façon synchrone et répond quand l'indexation est terminée.
     * Réponse attendue : {"status": "success", "message": "N chunk(s) ingéré(s) depuis URL"}
     */
    public Mono<Map<String, Object>> ingestWebPages() {
        return fastApiWebClient.post()
                .uri("/ingestion/web-pages")
                .retrieve()
                .bodyToMono(MAP_TYPE)
                .onErrorResume(e -> {
                    log.error("ingestWebPages failed: {}", e.getMessage());
                    return Mono.empty();
                });
    }

    public Mono<Map<String, Object>> deleteChunks(String source) {
        return fastApiWebClient.method(HttpMethod.DELETE)
                .uri("/ingestion/documents/by-source")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("source", source))
                .retrieve()
                .bodyToMono(MAP_TYPE)
                .onErrorResume(e -> {
                    log.warn("deleteChunks failed pour '{}': {}", source, e.getMessage());
                    return Mono.empty();
                });
    }

    public Mono<Map<String, Object>> ingestUrl(String url, List<String> cssClasses) {
        Map<String, Object> body = Map.of(
                "url", url,
                "classes", cssClasses != null ? cssClasses : List.of()
        );

        return fastApiWebClient.post()
                .uri("/ingestion/url")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(MAP_TYPE);
    }
}
