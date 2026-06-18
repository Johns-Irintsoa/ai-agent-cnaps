package mg.cnaps.admin.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {

    @Bean
    public WebClient fastApiWebClient(@Value("${rag.fastapi-url}") String fastApiUrl) {
        return WebClient.builder()
                .baseUrl(fastApiUrl)
                .codecs(config -> config.defaultCodecs().maxInMemorySize(60 * 1024 * 1024))
                .build();
    }
}
