// ingest_check.js - For Vercel
import { Pinecone } from '@pinecone-database/pinecone';

export default async function handler(req, res) {
    try {
        const pc = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
            environment: process.env.PINECONE_ENVIRONMENT
        });
        
        const index = pc.index(process.env.PINECONE_INDEX_NAME);
        const stats = await index.describeIndexStats();
        
        const hasDocuments = stats.totalVectorCount > 0;
        
        res.status(200).json({
            hasDocuments,
            vectorCount: stats.totalVectorCount,
            status: hasDocuments ? 'ready' : 'empty',
            message: hasDocuments ? '✅ Documents loaded' : '❌ No documents found. Run ingest_all.py locally.'
        });
        
    } catch (error) {
        res.status(500).json({
            error: error.message,
            status: 'error'
        });
    }
}