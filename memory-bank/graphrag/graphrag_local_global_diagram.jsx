import React, { useState } from 'react';

const GraphRAGDiagram = () => {
  const [activeTab, setActiveTab] = useState('local');
  const [hoveredNode, setHoveredNode] = useState(null);

  // Sample graph nodes representing entities
  const nodes = [
    { id: 1, x: 200, y: 100, label: 'John', community: 'A' },
    { id: 2, x: 300, y: 80, label: 'MIT', community: 'A' },
    { id: 3, x: 280, y: 180, label: 'AI Lab', community: 'A' },
    { id: 4, x: 150, y: 200, label: 'Paper X', community: 'A' },
    { id: 5, x: 450, y: 120, label: 'Mary', community: 'B' },
    { id: 6, x: 520, y: 200, label: 'Stanford', community: 'B' },
    { id: 7, x: 400, y: 250, label: 'NLP', community: 'B' },
    { id: 8, x: 350, y: 350, label: 'GraphRAG', community: 'C' },
    { id: 9, x: 250, y: 320, label: 'Microsoft', community: 'C' },
    { id: 10, x: 450, y: 380, label: 'Knowledge Graphs', community: 'C' },
  ];

  const edges = [
    { from: 1, to: 2 }, { from: 1, to: 3 }, { from: 1, to: 4 },
    { from: 2, to: 3 }, { from: 5, to: 6 }, { from: 5, to: 7 },
    { from: 6, to: 7 }, { from: 7, to: 8 }, { from: 8, to: 9 },
    { from: 8, to: 10 }, { from: 3, to: 7 }, { from: 4, to: 8 },
  ];

  const communityColors = {
    'A': '#3b82f6',
    'B': '#10b981',
    'C': '#f59e0b'
  };

  const communitySummaries = {
    'A': 'Research community at MIT focusing on AI and machine learning papers',
    'B': 'NLP research group at Stanford with focus on language models',
    'C': 'Knowledge graph and RAG systems development at Microsoft'
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <h1 className="text-3xl font-bold text-center mb-2">GraphRAG: Local vs Global Search</h1>
      <p className="text-gray-400 text-center mb-6">Interactive visualization of search strategies</p>
      
      {/* Tab Navigation */}
      <div className="flex justify-center gap-4 mb-6">
        {['local', 'global', 'drift'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === tab
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)} Search
          </button>
        ))}
      </div>

      <div className="flex gap-6">
        {/* Graph Visualization */}
        <div className="flex-1 bg-gray-800 rounded-xl p-4">
          <svg width="100%" height="450" viewBox="0 0 600 450">
            {/* Community backgrounds */}
            {activeTab !== 'local' && (
              <>
                <ellipse cx="220" cy="150" rx="120" ry="100" fill="#3b82f620" stroke="#3b82f6" strokeWidth="2" strokeDasharray="5,5" />
                <ellipse cx="460" cy="180" rx="110" ry="120" fill="#10b98120" stroke="#10b981" strokeWidth="2" strokeDasharray="5,5" />
                <ellipse cx="350" cy="350" rx="130" ry="80" fill="#f59e0b20" stroke="#f59e0b" strokeWidth="2" strokeDasharray="5,5" />
                <text x="220" y="40" textAnchor="middle" fill="#3b82f6" className="text-sm">Community A</text>
                <text x="520" y="60" textAnchor="middle" fill="#10b981" className="text-sm">Community B</text>
                <text x="350" y="420" textAnchor="middle" fill="#f59e0b" className="text-sm">Community C</text>
              </>
            )}

            {/* Edges */}
            {edges.map((edge, idx) => {
              const from = nodes.find(n => n.id === edge.from);
              const to = nodes.find(n => n.id === edge.to);
              const isHighlighted = activeTab === 'local' && 
                (hoveredNode === edge.from || hoveredNode === edge.to);
              return (
                <line
                  key={idx}
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke={isHighlighted ? '#60a5fa' : '#4b5563'}
                  strokeWidth={isHighlighted ? 3 : 1.5}
                  className="transition-all duration-300"
                />
              );
            })}

            {/* Nodes */}
            {nodes.map((node) => {
              const isHovered = hoveredNode === node.id;
              const isConnected = activeTab === 'local' && edges.some(
                e => (e.from === hoveredNode && e.to === node.id) || 
                     (e.to === hoveredNode && e.from === node.id)
              );
              return (
                <g
                  key={node.id}
                  onMouseEnter={() => setHoveredNode(node.id)}
                  onMouseLeave={() => setHoveredNode(null)}
                  className="cursor-pointer"
                >
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={isHovered ? 28 : 22}
                    fill={communityColors[node.community]}
                    stroke={isHovered || isConnected ? '#fff' : 'transparent'}
                    strokeWidth={3}
                    className="transition-all duration-300"
                    opacity={activeTab === 'local' && !isHovered && !isConnected && hoveredNode ? 0.3 : 1}
                  />
                  <text
                    x={node.x}
                    y={node.y + 4}
                    textAnchor="middle"
                    fill="white"
                    fontSize="11"
                    fontWeight="bold"
                  >
                    {node.label}
                  </text>
                </g>
              );
            })}

            {/* Query indicator */}
            {activeTab === 'local' && hoveredNode && (
              <g>
                <circle
                  cx={nodes.find(n => n.id === hoveredNode)?.x}
                  cy={nodes.find(n => n.id === hoveredNode)?.y}
                  r={35}
                  fill="none"
                  stroke="#60a5fa"
                  strokeWidth={2}
                  strokeDasharray="4,4"
                >
                  <animate attributeName="r" values="35;45;35" dur="1.5s" repeatCount="indefinite" />
                </circle>
              </g>
            )}
          </svg>
        </div>

        {/* Info Panel */}
        <div className="w-80 bg-gray-800 rounded-xl p-5">
          {activeTab === 'local' && (
            <div>
              <h3 className="text-xl font-bold text-blue-400 mb-3">🔍 Local Search</h3>
              <div className="space-y-3 text-sm text-gray-300">
                <p><span className="text-blue-400 font-semibold">Best for:</span> Specific entity queries</p>
                <p><span className="text-blue-400 font-semibold">Example:</span> "Who is John and what are his relationships?"</p>
                <div className="bg-gray-700 rounded-lg p-3 mt-4">
                  <p className="font-semibold text-white mb-2">Process:</p>
                  <ol className="list-decimal list-inside space-y-1">
                    <li>Embed query as vector</li>
                    <li>Find semantically similar entities</li>
                    <li>Fan out to neighbors (1-2 hops)</li>
                    <li>Retrieve relationships + text</li>
                    <li>Generate focused answer</li>
                  </ol>
                </div>
                <p className="text-green-400 mt-3">✓ Fast & low cost</p>
                <p className="text-yellow-400">⚡ Hover over nodes to see local traversal</p>
              </div>
            </div>
          )}
          
          {activeTab === 'global' && (
            <div>
              <h3 className="text-xl font-bold text-green-400 mb-3">🌐 Global Search</h3>
              <div className="space-y-3 text-sm text-gray-300">
                <p><span className="text-green-400 font-semibold">Best for:</span> Thematic, corpus-wide questions</p>
                <p><span className="text-green-400 font-semibold">Example:</span> "What are the main research themes?"</p>
                <div className="bg-gray-700 rounded-lg p-3 mt-4">
                  <p className="font-semibold text-white mb-2">Process (Map-Reduce):</p>
                  <ol className="list-decimal list-inside space-y-1">
                    <li>Select community hierarchy level</li>
                    <li>MAP: Each community generates partial answer</li>
                    <li>Score relevance of each partial</li>
                    <li>REDUCE: Aggregate all partials</li>
                    <li>Generate comprehensive response</li>
                  </ol>
                </div>
                <div className="mt-4 space-y-2">
                  <p className="font-semibold text-white">Community Summaries:</p>
                  {Object.entries(communitySummaries).map(([key, val]) => (
                    <div key={key} className="flex items-start gap-2">
                      <div className="w-3 h-3 rounded-full mt-1" style={{backgroundColor: communityColors[key]}}></div>
                      <p className="text-xs">{val}</p>
                    </div>
                  ))}
                </div>
                <p className="text-yellow-400 mt-3">⚠ Higher token cost</p>
              </div>
            </div>
          )}
          
          {activeTab === 'drift' && (
            <div>
              <h3 className="text-xl font-bold text-purple-400 mb-3">🔄 DRIFT Search</h3>
              <div className="space-y-3 text-sm text-gray-300">
                <p><span className="text-purple-400 font-semibold">Best for:</span> Complex queries needing both breadth & depth</p>
                <p className="text-xs italic">(Dynamic Reasoning and Inference with Flexible Traversal)</p>
                <div className="bg-gray-700 rounded-lg p-3 mt-4">
                  <p className="font-semibold text-white mb-2">Three-Phase Process:</p>
                  <div className="space-y-2">
                    <div className="bg-blue-900/50 rounded p-2">
                      <p className="font-semibold text-blue-300">1. PRIMER (Global)</p>
                      <p className="text-xs">Compare query with top-K community reports, generate initial answer + follow-ups</p>
                    </div>
                    <div className="bg-green-900/50 rounded p-2">
                      <p className="font-semibold text-green-300">2. FOLLOW-UP (Local)</p>
                      <p className="text-xs">Refine via local search iterations, producing intermediate answers</p>
                    </div>
                    <div className="bg-yellow-900/50 rounded p-2">
                      <p className="font-semibold text-yellow-300">3. OUTPUT</p>
                      <p className="text-xs">Ranked hierarchy of Q&A pairs by relevance</p>
                    </div>
                  </div>
                </div>
                <p className="text-green-400 mt-3">✓ 78% better than local search alone</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Legend & Stats */}
      <div className="mt-6 bg-gray-800 rounded-xl p-4 flex justify-between items-center">
        <div className="flex gap-6">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-500"></div>
            <span className="text-sm">Community A (MIT/AI)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-green-500"></div>
            <span className="text-sm">Community B (Stanford/NLP)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-yellow-500"></div>
            <span className="text-sm">Community C (Microsoft/KG)</span>
          </div>
        </div>
        <div className="text-sm text-gray-400">
          <span className="text-white font-semibold">Microsoft GraphRAG</span> | 29K+ ⭐ | arXiv:2404.16130
        </div>
      </div>
    </div>
  );
};

export default GraphRAGDiagram;
