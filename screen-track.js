import screenshot from 'screenshot-desktop';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Anthropic } from '@anthropic-ai/sdk';
import dotenv from 'dotenv';
// import { VectorStore } from 'vectorstore';
import readline from 'readline';
import { parse } from 'date-fns';
import { pipeline } from '@xenova/transformers';

// Helper function to convert filename-style timestamps to ISO 8601
function convertFilenameTimestampToISO(ts) {
  if (typeof ts !== 'string') return ts; // Return as is if not a string

  // If it's already a valid ISO that Date can parse, and contains T, colons, and a period (or Z at the end)
  // This is a basic check; a more robust ISO validation could be used if needed.
  if (ts.includes('T') && ts.includes(':') && (ts.includes('.') || ts.endsWith('Z')) && !isNaN(new Date(ts).getTime())) {
    return ts; // Assume it's already in a good, parseable ISO format
  }

  // Expected input like: 2025-05-11T21-45-09-471Z
  // Desired output: 2025-05-11T21:45:09.471Z
  const parts = ts.split('T');
  if (parts.length !== 2) {
    // console.warn(`Timestamp ${ts} is not in expected YYYY-MM-DDTHH-MM-SS-FFFZ format for conversion.`);
    return ts; // Return original if not in expected parts
  }

  let timePart = parts[1];
  const zSuffix = timePart.endsWith('Z') ? 'Z' : '';
  if (zSuffix) {
    timePart = timePart.slice(0, -1); // Remove Z for processing
  }

  const timeSegments = timePart.split('-');
  if (timeSegments.length === 4) { // HH-MM-SS-FFF
    return `${parts[0]}T${timeSegments[0]}:${timeSegments[1]}:${timeSegments[2]}.${timeSegments[3]}${zSuffix}`;
  } else if (timeSegments.length === 3) { // HH-MM-SS (no milliseconds)
    return `${parts[0]}T${timeSegments[0]}:${timeSegments[1]}:${timeSegments[2]}${zSuffix}`;
  }

  // console.warn(`Timestamp ${ts} could not be converted to ISO format.`);
  return ts; // Fallback: return original if format is not matched
}

// Helper function for cosine similarity
function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length || vecA.length === 0) {
    // console.warn('Cosine similarity: Invalid vectors or zero length.', vecA, vecB);
    return 0;
  }
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) {
    // console.warn('Cosine similarity: Zero norm vector.');
    return 0;
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Self-contained InMemoryVectorStore class
class InMemoryVectorStore {
  constructor(embedderInstance) {
    if (!embedderInstance || typeof embedderInstance.embedQuery !== 'function' || typeof embedderInstance.embedDocuments !== 'function') {
        throw new Error('InMemoryVectorStore requires an embedder with embedQuery and embedDocuments methods.');
    }
    this.embedder = embedderInstance;
    this.vectors = []; // Stores { id, embedding, document, metadata }
    console.log('Custom InMemoryVectorStore initialized.');
  }

  async add(item) { // item: { id, document, metadata }
    try {
      // Using embedDocuments as it's generally robust for single/multiple items.
      // Ensure your embedder.embedDocuments returns an array of embeddings.
      const embeddings = await this.embedder.embedDocuments([item.document]);
      if (!embeddings || embeddings.length === 0 || !embeddings[0]) {
          console.error('Embedding failed or returned no result for document:', item.document);
          throw new Error('Embedding failed or returned no result.');
      }
      const embedding = embeddings[0];
      
      this.vectors.push({
        id: item.id,
        embedding: embedding,
        document: item.document,
        metadata: item.metadata,
      });
      // console.log(`InMemoryVectorStore: Added item ${item.id}`);
    } catch (error) {
      console.error(`InMemoryVectorStore add error for id ${item.id}:`, error.message, error.stack);
      // Depending on desired behavior, you might want to re-throw or handle more gracefully
    }
  }

  async query(queryOptions) { // queryOptions: { queryText, nResults, where (optional) }
    try {
      const queryEmbedding = await this.embedder.embedQuery(queryOptions.queryText);
      if (!queryEmbedding) {
          console.error('Query embedding failed for text:', queryOptions.queryText);
          throw new Error('Query embedding failed.');
      }

      const scoredVectors = this.vectors.map(vec => {
        if (!vec.embedding) {
            console.warn(`Vector with id ${vec.id} has no embedding. Skipping.`);
            return { ...vec, similarity: -1 }; // Or filter out later
        }
        return {
          ...vec,
          similarity: cosineSimilarity(queryEmbedding, vec.embedding),
        }
      }).filter(vec => vec.similarity > -1); // Filter out vectors that had no embedding

      scoredVectors.sort((a, b) => b.similarity - a.similarity);

      let results = scoredVectors;
      if (queryOptions.where && typeof queryOptions.where === 'function') {
        results = results.filter(vec => queryOptions.where(vec.metadata));
      }
      
      return results.slice(0, queryOptions.nResults || 5).map(vec => ({
        document: vec.document,
        metadata: vec.metadata,
        // similarity: vec.similarity // Optional: useful for debugging
      }));
    } catch (error) {
      console.error('InMemoryVectorStore query error:', error.message, error.stack);
      return []; // Return empty array on error
    }
  }
}

// Get current file's directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialize environment variables
dotenv.config();

// Initialize Anthropic client
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// Custom embedding function using @xenova/transformers
let embedder;
async function initializeEmbedder() {
  try {
    const pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    embedder = {
      async embedDocuments(texts) {
        const embeddings = [];
        for (const text of texts) {
          const output = await pipe(text, { pooling: 'mean', normalize: true });
          embeddings.push(Array.from(output.data));
        }
        return embeddings;
      },
      async embedQuery(text) {
        const output = await pipe(text, { pooling: 'mean', normalize: true });
        return Array.from(output.data);
      },
    };
    console.log('Embedder initialized successfully');
  } catch (error) {
    console.error('Embedder initialization error:', error.message);
    process.exit(1);
  }
}

// Initialize vector store (in-memory)
// let vectorStore;
// Initialize vector store (in-memory)
let vectorStore; // Keep this global variable
async function initializeVectorStore() {
  try {
    if (!embedder) { // 'embedder' is your global variable initialized by initializeEmbedder()
      console.error('Embedder not initialized before vector store. Make sure initializeEmbedder() is called and completes first.');
      return null;
    }
    // Use the new InMemoryVectorStore and pass the initialized embedder
    vectorStore = new InMemoryVectorStore(embedder); 
    console.log('Vector store initialized successfully using InMemoryVectorStore');
    return vectorStore;
  } catch (error) {
    console.error('Vector store (custom in-memory) initialization error:', error.message);
    return null;
  }
}

// Ensure directories exist
async function ensureDirectories() {
  const screenshotsDir = path.join(__dirname, 'screenshots');
  const screenhistoryDir = path.join(__dirname, 'screenhistory');
  
  for (const dir of [screenshotsDir, screenhistoryDir]) {
    try {
      await fs.mkdir(dir, { recursive: true });
    } catch (err) {
      if (err.code !== 'EEXIST') throw err;
    }
  }
  
  return { screenshotsDir, screenhistoryDir };
}

// Convert buffer to base64
function bufferToBase64(buffer) {
  return buffer.toString('base64');
}

// Get current timestamp in ISO format (for filenames)
function getTimestampForFilename() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

// Process screenshot with Claude
async function processScreenshotWithClaude(imageBase64) {
  try {
    console.log('Sending request to Claude...');
    const response = await anthropic.messages.create({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 1000,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze this screenshot and provide a detailed description in the following JSON format: { "active_app": "", "summary": "", "extracted_text": "", "task_category": "", "productivity_score": 0, "workflow_suggestions": "" }'
            },
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/jpeg',
                data: imageBase64
              }
            }
          ]
        }
      ]
    });

    const cleanedResponse = response.content[0].text
      .replace(/```json\n?/g, '')
      .replace(/```\n?/g, '')
      .trim();
    
    return JSON.parse(cleanedResponse);
  } catch (error) {
    console.error('Claude API Error:', error.message);
    return null;
  }
}

// Add JSON data to vector store
async function addToVectorStore(collection, analysisData) { // analysisData should have analysisData.timestamp as ISO
  try {
    const textToEmbed = `${analysisData.summary} ${analysisData.extracted_text}`;
    
    // Ensure analysisData.timestamp is a valid ISO string
    if (!analysisData.timestamp || isNaN(new Date(analysisData.timestamp).getTime())) {
      console.error(`Invalid or missing timestamp in analysisData for addToVectorStore. Timestamp: ${analysisData.timestamp}. Data:`, analysisData);
      return; 
    }

    await collection.add({
      id: analysisData.timestamp, // Use the ISO timestamp from analysisData as ID
      document: textToEmbed,
      metadata: { ...analysisData }, // analysisData itself becomes metadata, includes its .timestamp
    });
    // console.log(`Added to vector store (ID: ${analysisData.timestamp})`);
  } catch (error) {
    // console.error(`Vector store add error for ID ${analysisData.timestamp}:`, error.message);
  }
}

// Parse natural language time references (existing simple parser)
function parseTimeQuery(queryText) {
  const now = new Date();
  let startTime, endTime;

  if (queryText.toLowerCase().includes('yesterday')) {
    startTime = new Date(now);
    startTime.setDate(now.getDate() - 1);
    startTime.setHours(0, 0, 0, 0);
    endTime = new Date(now);
    endTime.setDate(now.getDate() - 1);
    endTime.setHours(23, 59, 59, 999);
  } else if (queryText.match(/\d{4}-\d{2}-\d{2}/)) {
    const dateStr = queryText.match(/\d{4}-\d{2}-\d{2}/)[0];
    try {
        // Attempt to parse with date-fns, assuming the matched string is the date
        startTime = parse(dateStr, 'yyyy-MM-dd', new Date());
        startTime.setHours(0, 0, 0, 0);
        endTime = new Date(startTime);
        endTime.setHours(23, 59, 59, 999);
    } catch (e) {
        console.warn(`Could not parse date string: ${dateStr} with date-fns. Falling back or skipping.`);
        return null;
    }
  } else {
    return null;
  }

  return {
    start: startTime.toISOString(),
    end: endTime.toISOString(),
  };
}

// New function to parse natural language time with Claude
async function parseNaturalLanguageTimeWithClaude(queryText) {
  try {
    const currentISODate = new Date().toISOString();
    const prompt = `Given the user's query: "${queryText}"

Analyze this query to identify any specific dates, date ranges, or relative time references (like "today", "yesterday", "last Tuesday", "this week", "last month", "between May 1st and May 5th").

If a time reference is found, provide the start and end of that time range in ISO 8601 format (YYYY-MM-DDTHH:mm:ss.sssZ).

For "today", the range is from the beginning of today to the end of today.
For "yesterday", from the beginning of yesterday to the end of yesterday.
For "this week", assume the week starts on Monday, provide the range from the beginning of this Monday to the end of this coming Sunday.
For "last month", provide the range for the entire previous calendar month.
If a single date is mentioned (e.g., "on May 10th"), provide the range for that entire day.
If an open-ended range like "since Monday" is mentioned, use the current time as the end of the range.
If no specific time reference is found, or if it's too vague, output null for startTimeISO and endTimeISO.

Current date for reference: ${currentISODate}

Output ONLY the JSON object like this:
{
"startTimeISO": "YYYY-MM-DDTHH:mm:ss.sssZ_or_null",
"endTimeISO": "YYYY-MM-DDTHH:mm:ss.sssZ_or_null",
"cleanedQuery": "The user query with the time phrases removed or normalized, focusing on the core activity."
}
If no time is found, both startTimeISO and endTimeISO should be null, and cleanedQuery should be the original query.`;

    console.log("Requesting time parsing from Claude for query:", queryText);
    const response = await anthropic.messages.create({
      model: 'claude-3-opus-20240229', // Or your preferred model, Opus might be better for complex instructions
      max_tokens: 300,
      messages: [{ role: 'user', content: prompt }],
    });

    const resultText = response.content[0].text.trim();
    console.log("Claude time parsing raw response:", resultText);
    // Robust JSON parsing
    try {
        // Remove potential markdown fences if Claude adds them
        const cleanedJsonString = resultText.replace(/^```json\s*|```$/g, '');
        const parsedResult = JSON.parse(cleanedJsonString);
        
        // Log the parsed result from Claude
        console.log("Claude parsed time and query:", parsedResult);

        if (parsedResult && parsedResult.cleanedQuery) {
             // Basic validation
            if ((parsedResult.startTimeISO && !parsedResult.endTimeISO) || (!parsedResult.startTimeISO && parsedResult.endTimeISO)) {
                console.warn("Claude returned partial time range, ignoring time filter for safety:", parsedResult);
                return { start: null, end: null, cleanedQuery: queryText }; // Fallback
            }
            // Validate ISO strings before returning
            const isValidISODate = (isoString) => {
                if (!isoString) return true; // null is valid
                return !isNaN(new Date(isoString).getTime());
            };

            if (!isValidISODate(parsedResult.startTimeISO) || !isValidISODate(parsedResult.endTimeISO)) {
                console.warn("Claude returned invalid ISO date format, ignoring time filter:", parsedResult);
                return { start: null, end: null, cleanedQuery: queryText }; // Fallback
            }

            return {
                start: parsedResult.startTimeISO || null, // Ensure null if empty string or undefined
                end: parsedResult.endTimeISO || null,   // Ensure null if empty string or undefined
                cleanedQuery: parsedResult.cleanedQuery || queryText // Fallback for cleaned query
            };
        } else {
            console.warn("Claude response for time parsing did not contain cleanedQuery or was not as expected:", parsedResult);
        }
    } catch (e) {
      console.error('Error parsing time extraction response from Claude:', e, "Raw response:", resultText);
    }
  } catch (error) {
    console.error('Claude API Error during time parsing:', error.message);
  }
  // Fallback if Claude parsing fails or returns unexpected format
  console.log("Falling back to original query due to Claude time parsing issue.");
  return { start: null, end: null, cleanedQuery: queryText };
}

// Query vector store and generate response with Claude - UPDATED
async function queryVectorStore(collection, originalQueryText) {
  try {
    console.log(`Original user query: "${originalQueryText}"`);
    // Attempt to parse time using Claude
    const timeParseResult = await parseNaturalLanguageTimeWithClaude(originalQueryText);
    
    let timeFilter = null;
    if (timeParseResult && timeParseResult.start && timeParseResult.end) {
      // Validate that start and end are valid date strings before creating Date objects
      const startDate = new Date(timeParseResult.start);
      const endDate = new Date(timeParseResult.end);
      if (!isNaN(startDate.getTime()) && !isNaN(endDate.getTime())) {
        timeFilter = {
          start: startDate.toISOString(),
          end: endDate.toISOString(),
        };
      } else {
        console.warn(`Invalid date strings from Claude: start='${timeParseResult.start}', end='${timeParseResult.end}'. Proceeding without time filter.`);
      }
    }
    
    // Use the cleaned query from Claude's time parsing, or the original if parsing failed/no time found
    let queryForEmbedding = (timeParseResult && timeParseResult.cleanedQuery) ? timeParseResult.cleanedQuery : originalQueryText;
    if (!queryForEmbedding.trim()) { // If cleaning removed everything, use a generic query
        queryForEmbedding = 'What was I doing?'; // Default query if cleaning results in empty string
    }

    // Log the query to be used for embedding and the time filter
    console.log(`Executing vector query with: Cleaned Query='${queryForEmbedding}', StartTime='${timeFilter ? timeFilter.start : 'N/A'}', EndTime='${timeFilter ? timeFilter.end : 'N/A'}'`);

    let results;
    if (timeFilter) {
      results = await collection.query({
        queryText: queryForEmbedding,
        nResults: 10, // Potentially fetch more if time filtering is strict
        where: (metadata) => {
          if (!metadata.timestamp) return false;
          try {
            const metadataTimestamp = new Date(metadata.timestamp).toISOString();
            return metadataTimestamp >= timeFilter.start && metadataTimestamp <= timeFilter.end;
          } catch (e) {
            console.warn(`Could not parse metadata.timestamp: ${metadata.timestamp}`);
            return false;
          }
        }
      });
    } else {
      // Fallback: If Claude doesn't provide a time filter, try the original simple parser
      const simpleTimeFilter = parseTimeQuery(originalQueryText); 
      if (simpleTimeFilter) {
        console.log("Using fallback simple time parser. Filter:", simpleTimeFilter);
        // Attempt to clean the query text based on what simpleTimeFilter might have parsed
        let fallbackQueryForEmbedding = originalQueryText
            .replace(/yesterday/gi, '') 
            .replace(/\d{4}-\d{2}-\d{2}/, '') 
            .trim();
        if (!fallbackQueryForEmbedding) fallbackQueryForEmbedding = 'What was I doing?';
        
        console.log(`Executing vector query with (fallback time filter): Cleaned Query='${fallbackQueryForEmbedding}', StartTime='${simpleTimeFilter.start}', EndTime='${simpleTimeFilter.end}'`);
        results = await collection.query({
            queryText: fallbackQueryForEmbedding,
            nResults: 5,
            where: (metadata) => {
                if (!metadata.timestamp) return false;
                 try {
                    const metadataTimestamp = new Date(metadata.timestamp).toISOString();
                    return metadataTimestamp >= simpleTimeFilter.start && metadataTimestamp <= simpleTimeFilter.end;
                } catch (e) {
                    console.warn(`Could not parse metadata.timestamp: ${metadata.timestamp}`);
                    return false;
                }
            }
        });
      } else {
        // No time filter from Claude or simple parser
        console.log(`Executing vector query with: Cleaned Query='${queryForEmbedding}' (no time filter)`);
        results = await collection.query({
            queryText: queryForEmbedding,
            nResults: 5,
        });
      }
    }

    const context = results.map((result) => ({
      document: result.document,
      metadata: result.metadata,
    }));

    console.log('Retrieved context from vector store (RAG results):', JSON.stringify(context, null, 2));

    if (context.length === 0) {
        return "I couldn't find any activity matching your query and time range.";
    }

    const promptForClaudeSummary = `Based on the following context, answer the query: "${originalQueryText}"

Context:
${context.map((c, i) => `Document ${i + 1}: ${c.document} (Timestamp: ${c.metadata.timestamp})`).join('\n')}

Answer in a concise, natural language format. If the query asks for a summary of activities, provide that.`;
    
    console.log("Sending context to Claude for final summarization. Original query:", originalQueryText);
    const response = await anthropic.messages.create({
      model: 'claude-3-5-sonnet-20240620', 
      max_tokens: 500,
      messages: [{ role: 'user', content: promptForClaudeSummary }],
    });

    return response.content[0].text;

  } catch (error) {
    console.error('Query error in queryVectorStore:', error.message, error.stack);
    return 'Error processing query.';
  }
}

// Terminal query interface
function startQueryInterface(collection) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  rl.setPrompt('Query> ');
  rl.prompt();

  rl.on('line', async (line) => {
    if (line.trim().toLowerCase() === 'exit') {
      rl.close();
      return;
    }

    console.log('Processing query:', line);
    const response = await queryVectorStore(collection, line);
    console.log('Response:', response);
    rl.prompt();
  });

  rl.on('close', () => {
    console.log('Query interface closed.');
    process.exit(0);
  });
}

// Add this function or integrate its logic into trackScreen

async function loadExistingHistory(collection, screenhistoryDir) {
  console.log('Attempting to load existing screen history...');
  try {
    const files = await fs.readdir(screenhistoryDir);
    const jsonFiles = files.filter(file => path.extname(file).toLowerCase() === '.json');
    console.log(`Found ${jsonFiles.length} JSON files in ${screenhistoryDir}.`);

    for (const file of jsonFiles) {
      const filePath = path.join(screenhistoryDir, file);
      try {
        const fileContent = await fs.readFile(filePath, 'utf-8');
        let analysis = null; // Initialize analysis
        if (fileContent.trim() === 'null' || fileContent.trim() === '') {
            analysis = null;
        } else {
            try {
                analysis = JSON.parse(fileContent);
            } catch (parseError) {
                console.warn(`Skipping ${file}: Could not parse JSON. Error: ${parseError.message}`);
                analysis = null; // Ensure analysis is null if parsing fails
            }
        }

        // MODIFIED CHECK HERE:
        if (analysis && typeof analysis.summary === 'string' && typeof analysis.extracted_text === 'string') {
          // Convert timestamp from file (potentially old format or new) to standard ISO
          let isoTimestamp = analysis.timestamp;
          
          // Attempt conversion if it looks like the old format or isn't directly parsable
          if (isoTimestamp && typeof isoTimestamp === 'string') {
            if (isNaN(new Date(isoTimestamp).getTime()) || ((isoTimestamp.match(/-/g) || []).length > 2 && !isoTimestamp.includes(':'))) {
                // console.log(`Old format detected or unparsable, attempting conversion for: ${isoTimestamp}`);
                isoTimestamp = convertFilenameTimestampToISO(analysis.timestamp);
            }
          }
          
          // Validate the timestamp after potential conversion
          if (!isoTimestamp || isNaN(new Date(isoTimestamp).getTime())) {
            console.warn(`Skipping file ${file}: Timestamp '${analysis.timestamp}' (raw) resulted in invalid ISO timestamp '${isoTimestamp}' after potential conversion.`);
            continue;
          }
          
          analysis.timestamp = isoTimestamp; // Update analysis object with the correct ISO timestamp

          await addToVectorStore(collection, analysis);
        } else {
          console.warn(`Skipping ${file}: missing essential data (timestamp, summary, or extracted_text).`);
        }
      } catch (err) {
        console.error(`Error processing history file ${file}: ${err.message}`);
      }
    }
    console.log('Finished loading existing screen history into vector store.');
  } catch (err) {
    console.error(`Error reading screenhistory directory ${screenhistoryDir}: ${err.message}`);
  }
}


// Main tracking function
async function trackScreen() {
  try {
    console.log('Starting screen tracking...');
    await initializeEmbedder(); // Initialize transformer model
    const { screenshotsDir, screenhistoryDir } = await ensureDirectories();
    const collection = await initializeVectorStore();

    if (!collection) {
      throw new Error('Failed to initialize vector store');
    }

    await loadExistingHistory(collection, screenhistoryDir);

    // Start query interface in parallel
    startQueryInterface(collection);

    // Main loop for capturing new screenshots
    while (true) {
      const currentISOTimestamp = new Date().toISOString(); // Standard ISO 8601 timestamp
      const filenameTimestamp = currentISOTimestamp.replace(/[:.]/g, '-'); // For filename compatibility

      const screenshotPath = path.join(screenshotsDir, `${filenameTimestamp}.jpg`);
      const jsonPath = path.join(screenhistoryDir, `${filenameTimestamp}.json`);
      
      // console.log('Capturing screenshot...');
      const img = await screenshot();
      await fs.writeFile(screenshotPath, img);
      // console.log(`Screenshot saved to ${screenshotPath}`);
      
      const base64Image = bufferToBase64(img);
      const analysis = await processScreenshotWithClaude(base64Image);
      
      if (analysis) {
        analysis.timestamp = currentISOTimestamp; // Store the standard ISO timestamp in the analysis object
        await fs.writeFile(jsonPath, JSON.stringify(analysis, null, 2));
        // console.log(`Analysis saved to ${jsonPath}`);
        await addToVectorStore(collection, analysis); // Pass the analysis object
      } else {
        // console.log('Analysis from Claude was null, skipping add to vector store.');
      }
      
      const waitTimeSeconds = 60;
      // console.log(`Waiting ${waitTimeSeconds} seconds before next screenshot...`);
      await new Promise(resolve => setTimeout(resolve, waitTimeSeconds * 1000));
    }
  } catch (error) {
    console.error('Error in screen tracking:', error);
    // Optional: implement a more robust retry or exit strategy
    console.log('Restarting trackScreen in 5 seconds due to error...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    trackScreen(); // Be cautious with recursive calls like this without a backoff or limit
  }
}

// Start tracking
console.log(`Script boot time: ${new Date().toISOString()}`); // Boot time logging
console.log('Initializing screen tracking script...');
trackScreen().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});