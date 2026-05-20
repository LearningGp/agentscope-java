/*
 * Copyright 2024-2026 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.agentscope.core.rag;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.AgentBase;
import io.agentscope.core.hook.PreCallEvent;
import io.agentscope.core.interruption.InterruptContext;
import io.agentscope.core.memory.InMemoryMemory;
import io.agentscope.core.memory.LongTermMemory;
import io.agentscope.core.memory.StaticLongTermMemoryHook;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.rag.model.Document;
import io.agentscope.core.rag.model.DocumentMetadata;
import io.agentscope.core.rag.model.RetrieveConfig;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

/** Unit tests for {@link GenericRAGHook}. */
class GenericRAGHookTest {

    private Agent mockAgent;

    @BeforeEach
    void setUp() {
        mockAgent = createMockAgent("TestAgent");
    }

    private Agent createMockAgent(String name) {
        return new AgentBase(name) {

            @Override
            protected Mono<Msg> doCall(List<Msg> msgs) {
                return Mono.just(msgs.get(0));
            }

            @Override
            protected Mono<Void> doObserve(Msg msg) {
                return Mono.empty();
            }

            @Override
            protected Mono<Msg> handleInterrupt(InterruptContext context, Msg... originalArgs) {
                return Mono.just(
                        Msg.builder()
                                .name(getName())
                                .role(MsgRole.ASSISTANT)
                                .content(TextBlock.builder().text("Interrupted").build())
                                .build());
            }
        };
    }

    @Test
    void testLongTermMemoryThenRagUsesOriginalUserQuery() {
        LongTermMemory longTermMemory = mock(LongTermMemory.class);
        StaticLongTermMemoryHook memoryHook =
                new StaticLongTermMemoryHook(longTermMemory, new InMemoryMemory());
        CapturingKnowledge knowledge = new CapturingKnowledge();
        GenericRAGHook ragHook = new GenericRAGHook(knowledge);
        PreCallEvent event = new PreCallEvent(mockAgent, List.of(userMessage("Original query")));

        when(longTermMemory.retrieve(any(Msg.class))).thenReturn(Mono.just("Remembered context"));

        StepVerifier.create(memoryHook.onEvent(event)).expectNext(event).verifyComplete();
        StepVerifier.create(ragHook.onEvent(event))
                .assertNext(
                        resultEvent -> {
                            List<Msg> messages = resultEvent.getInputMessages();
                            assertEquals(3, messages.size());
                            assertEquals("long_term_memory", messages.get(1).getName());
                            assertEquals("retrieved_knowledge", messages.get(2).getName());
                        })
                .verifyComplete();

        assertEquals("Original query", knowledge.getLastQuery());
    }

    @Test
    void testRagThenLongTermMemoryUsesOriginalUserQuery() {
        CapturingKnowledge knowledge = new CapturingKnowledge();
        GenericRAGHook ragHook = new GenericRAGHook(knowledge);
        LongTermMemory longTermMemory = mock(LongTermMemory.class);
        StaticLongTermMemoryHook memoryHook =
                new StaticLongTermMemoryHook(longTermMemory, new InMemoryMemory());
        PreCallEvent event = new PreCallEvent(mockAgent, List.of(userMessage("Original query")));

        when(longTermMemory.retrieve(any(Msg.class))).thenReturn(Mono.just("Remembered context"));

        StepVerifier.create(ragHook.onEvent(event)).expectNext(event).verifyComplete();
        StepVerifier.create(memoryHook.onEvent(event))
                .assertNext(
                        resultEvent -> {
                            List<Msg> messages = resultEvent.getInputMessages();
                            assertEquals(3, messages.size());
                            assertEquals("retrieved_knowledge", messages.get(1).getName());
                            assertEquals("long_term_memory", messages.get(2).getName());
                        })
                .verifyComplete();

        ArgumentCaptor<Msg> captor = ArgumentCaptor.forClass(Msg.class);
        verify(longTermMemory, times(1)).retrieve(captor.capture());
        assertEquals("Original query", captor.getValue().getTextContent());
    }

    @Test
    void testGenericRagHookAloneRetrievesAndInjectsKnowledge() {
        CapturingKnowledge knowledge = new CapturingKnowledge();
        GenericRAGHook ragHook = new GenericRAGHook(knowledge);
        List<Msg> inputMessages =
                List.of(
                        userMessage("First user message"),
                        Msg.builder()
                                .role(MsgRole.ASSISTANT)
                                .content(TextBlock.builder().text("Assistant reply").build())
                                .build(),
                        userMessage("Latest user query"));
        PreCallEvent event = new PreCallEvent(mockAgent, inputMessages);

        StepVerifier.create(ragHook.onEvent(event))
                .assertNext(
                        resultEvent -> {
                            List<Msg> messages = resultEvent.getInputMessages();
                            assertEquals(4, messages.size());
                            Msg injectedMessage = messages.get(3);
                            assertEquals(MsgRole.USER, injectedMessage.getRole());
                            assertEquals("retrieved_knowledge", injectedMessage.getName());
                            assertTrue(
                                    injectedMessage
                                            .getTextContent()
                                            .contains("<retrieved_knowledge>"));
                            assertTrue(injectedMessage.getTextContent().contains("Knowledge hit"));
                        })
                .verifyComplete();

        assertEquals("Latest user query", knowledge.getLastQuery());
    }

    private Msg userMessage(String text) {
        return Msg.builder()
                .role(MsgRole.USER)
                .content(TextBlock.builder().text(text).build())
                .build();
    }

    private static class CapturingKnowledge implements Knowledge {

        private final AtomicReference<String> lastQuery = new AtomicReference<>();

        @Override
        public Mono<Void> addDocuments(List<Document> documents) {
            return Mono.empty();
        }

        @Override
        public Mono<List<Document>> retrieve(String query, RetrieveConfig config) {
            lastQuery.set(query);
            Document document =
                    new Document(
                            new DocumentMetadata(
                                    TextBlock.builder().text("Knowledge hit").build(),
                                    "doc-1",
                                    "chunk-1"));
            document.setScore(0.9);
            return Mono.just(List.of(document));
        }

        String getLastQuery() {
            return lastQuery.get();
        }
    }
}
