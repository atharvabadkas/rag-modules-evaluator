import pandas as pd
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

class ApiKeyTest:
    def _create_rag_chain(self):
        """
        Create enhanced RAG chain with multi-dataset support and ChatGPT-like capabilities
        """
        try:
            documents = self.prepare_data_for_rag()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 30}  # Increased for better coverage
            )
            
            # Create dataset summaries with detailed capabilities
            dataset_summaries = []
            for name, metadata in self.dataset_manager.dataset_metadata.items():
                summary = f"- {name} Dataset ({metadata['type']}):\\n"
                summary += f"  * {metadata['rows']} records\\n"
                summary += f"  * Columns: {', '.join(metadata['columns'])}\\n"
                if metadata['date_columns']:
                    summary += f"  * Time-based analysis available on: {', '.join(metadata['date_columns'])}\\n"
                dataset_summaries.append(summary)
            
            prompt_template = f"""You are an advanced AI assistant specialized in data analysis and insights generation, similar to ChatGPT but focused on the available datasets. You can handle any type of query from simple lookups to complex multi-step analysis.

            Available Datasets and Capabilities:
            {chr(10).join(dataset_summaries)}

            Your capabilities include but are not limited to:
            1. Basic Information Retrieval
               - Finding specific data points
               - Filtering and counting records
               - Looking up values across datasets
            
            2. Statistical Analysis
               - Calculating averages, sums, medians, etc.
               - Finding patterns and outliers
               - Generating descriptive statistics
            
            3. Time-Series Analysis
               - Identifying trends over time
               - Analyzing seasonal patterns
               - Comparing different time periods
            
            4. Comparative Analysis
               - Comparing values across different categories
               - Cross-dataset analysis
               - Finding correlations and relationships
            
            5. Predictive Insights
               - Trend-based forecasting
               - Pattern recognition
               - What-if scenario analysis
            
            6. Complex Reasoning
               - Multi-step analysis
               - Cause-effect relationships
               - Business impact analysis
            
            7. Natural Language Responses
               - Explaining findings in clear language
               - Providing context and insights
               - Suggesting relevant follow-up analyses

            Context from the datasets:
            -------------------------
            {{context}}
            -------------------------

            Conversation History:
            {self._format_conversation_history()}

            Guidelines for Your Response:
            1. Always consider the ENTIRE dataset when performing calculations
            2. Provide specific numbers and statistics when relevant
            3. Explain your reasoning and methodology
            4. If you detect ambiguity in the question, explain your assumptions
            5. When appropriate, suggest related insights or follow-up analyses
            6. If the data is insufficient, explain what's missing
            7. For time-based queries, specify the time range used
            8. For complex analyses, break down your approach into steps
            
            Question: {{question}}
            
            Provide a comprehensive answer that includes:
            1. Direct answer to the question
            2. Supporting data and calculations
            3. Relevant context and insights
            4. Potential implications or recommendations
            5. Suggestions for further analysis if relevant

            Answer: """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    temperature=0.2,  # Slightly increased for more natural responses
                    model_name="gpt-3.5-turbo",
                    max_tokens=2000  # Increased for more detailed responses
                ),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PROMPT,
                    "verbose": True
                }
            )
            
            print("Advanced analysis system initialized and ready for queries")
            return chain
            
        except Exception as e:
            print(f"Error creating analysis chain: {str(e)}")
            return None

    def query(self, question: str) -> str:
        """
        Process a question with enhanced analysis capabilities
        """
        if self.chain is None:
            return "Error: System not properly initialized."
        
        try:
            print("\nAnalyzing query...")
            
            # Detect query type for enhanced processing
            query_indicators = {
                'time_series': ['trend', 'over time', 'pattern', 'monthly', 'weekly', 'daily', 'year', 'month'],
                'comparison': ['compare', 'difference', 'versus', 'vs', 'against', 'between'],
                'prediction': ['predict', 'forecast', 'future', 'next', 'expected', 'likely'],
                'aggregation': ['total', 'average', 'mean', 'median', 'sum', 'count', 'maximum', 'minimum'],
                'correlation': ['correlation', 'relationship', 'related', 'impact', 'effect', 'cause'],
                'distribution': ['distribution', 'spread', 'range', 'variance', 'deviation']
            }
            
            # Identify query types
            query_types = []
            question_lower = question.lower()
            for qtype, indicators in query_indicators.items():
                if any(ind in question_lower for ind in indicators):
                    query_types.append(qtype)
            
            # Get base response
            result = self.chain({"query": question})
            answer = result['result']
            sources = result['source_documents']
            
            # Add enhanced analysis based on query type
            additional_analysis = []
            
            # Time series analysis
            if 'time_series' in query_types:
                for dataset, metadata in self.dataset_manager.dataset_metadata.items():
                    if metadata['date_columns']:
                        for col in metadata['columns']:
                            if col.lower() in question_lower and pd.api.types.is_numeric_dtype(
                                self.dataset_manager.datasets[dataset][col].dtype):
                                analysis = self.analytics_engine.time_series_analysis(
                                    dataset, col, metadata['date_columns'][0])
                                if 'error' not in analysis:
                                    additional_analysis.append(
                                        f"\nDetailed Time Series Analysis for {col}:"
                                        f"\n- Overall Trend: {analysis['growth_rate']:.2%} growth rate"
                                        f"\n- Seasonality: {'Detected' if analysis['seasonality']['has_seasonality'] else 'Not detected'}"
                                        f"\n- Statistics: Mean={analysis['statistics']['mean']:.2f}, "
                                        f"Std={analysis['statistics']['std']:.2f}"
                                    )
            
            # Cross-dataset analysis
            if 'comparison' in query_types or 'correlation' in query_types:
                mentioned_datasets = [name for name in self.dataset_manager.datasets.keys() 
                                   if name.lower() in question_lower]
                if len(mentioned_datasets) >= 2:
                    for col in self.dataset_manager.datasets[mentioned_datasets[0]].columns:
                        if col.lower() in question_lower:
                            analysis = self.analytics_engine.cross_dataset_analysis(
                                mentioned_datasets[:2],
                                {d: col for d in mentioned_datasets[:2]}
                            )
                            if 'error' not in analysis and 'correlation' in analysis:
                                additional_analysis.append(
                                    f"\nCross-Dataset Analysis:"
                                    f"\n- Correlation between datasets: {analysis['correlation']:.2f}"
                                    f"\n- Comparison of {col}:"
                                    f"\n  * {mentioned_datasets[0]}: mean={analysis[mentioned_datasets[0]]['mean']:.2f}"
                                    f"\n  * {mentioned_datasets[1]}: mean={analysis[mentioned_datasets[1]]['mean']:.2f}"
                                )
            
            # Store in conversation history
            self.conversation_history.append((question, answer))
            
            # Format the response
            response = "\n" + "="*50
            response += f"\nAnalysis Results:\n{answer}\n"
            
            if additional_analysis:
                response += "\nAdditional Insights:"
                response += "\n".join(additional_analysis)
            
            if query_types:
                response += f"\n\nQuery Type(s) Detected: {', '.join(query_types)}"
            
            response += "\n\nSupporting Data:"
            for i, doc in enumerate(sources[:3], 1):
                response += f"\n{i}. {doc.page_content}"
            
            response += "\n" + "="*50
            
            return response
            
        except Exception as e:
            return f"Error processing query: {str(e)}"