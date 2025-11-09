import React, { useState } from 'react';
import { Send, User, FileText, Clock, ChevronLeft, ChevronRight, Sparkles } from 'lucide-react';

export default function PoliticalTransparencyUI() {
  const [chatMessages, setChatMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isChatExpanded, setIsChatExpanded] = useState(true);

  const handleSendMessage = () => {
    if (inputMessage.trim()) {
      setChatMessages([...chatMessages, { type: 'user', text: inputMessage }]);
      setInputMessage('');
      
      setTimeout(() => {
        setChatMessages(prev => [...prev, { 
          type: 'assistant', 
          text: 'I can help you understand political transparency. What would you like to know?' 
        }]);
      }, 1000);
    }
  };

  const suggestedQuestions = [
    "What bills has my representative voted on?",
    "Show me campaign promises vs actions",
    "How can I contact my representative?",
    "What are the latest updates?"
  ];

  const mockStories = [
    {
      id: 1,
      title: "Infrastructure Bill Vote Analysis",
      representative: "Sen. Jane Smith",
      alignment: 87,
      category: "Economy",
      summary: "Voted in favor of infrastructure spending, consistent with campaign promises on job creation.",
      timestamp: "2 hours ago"
    },
    {
      id: 2,
      title: "Healthcare Reform Statement",
      representative: "Rep. John Doe",
      alignment: 65,
      category: "Healthcare",
      summary: "Public statements support expansion, but recent committee votes show mixed record.",
      timestamp: "5 hours ago"
    },
    {
      id: 3,
      title: "Climate Policy Update",
      representative: "Sen. Maria Garcia",
      alignment: 92,
      category: "Environment",
      summary: "Strong alignment between campaign promises and legislative actions on renewable energy.",
      timestamp: "1 day ago"
    },
    {
      id: 4,
      title: "Education Funding Vote",
      representative: "Rep. Michael Chen",
      alignment: 78,
      category: "Education",
      summary: "Supported increased education funding, aligning with campaign trail commitments.",
      timestamp: "2 days ago"
    }
  ];

  return (
    <div className="flex h-screen bg-black text-white overflow-hidden">
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-gradient-to-r from-purple-900 to-black border-b border-purple-800/30 px-6 py-3">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Political Transparency</h1>
              <p className="text-xs text-purple-300">Track what they say vs. what they do</p>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto px-6 py-6 bg-gradient-to-b from-black to-purple-950/20">
          <div className="max-w-4xl mx-auto">
            {/* Welcome Section */}
            <div className="mb-8">
              <div className="flex items-center space-x-2 mb-3">
                <Sparkles className="w-6 h-6 text-purple-400" />
                <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Hey there!
                </h2>
              </div>
              <p className="text-lg text-gray-300">
                Here are some interesting stories for you today
              </p>
            </div>

            {/* Filter Tabs */}
            <div className="flex space-x-2 mb-6 overflow-x-auto pb-2 scrollbar-hide">
              {['All', 'Economy', 'Healthcare', 'Environment', 'Education', 'Justice'].map((category) => (
                <button
                  key={category}
                  className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all ${
                    category === 'All' 
                      ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/50' 
                      : 'bg-purple-900/30 text-purple-300 hover:bg-purple-800/40 border border-purple-700/30'
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>

            {/* News Feed */}
            <div className="space-y-4">
              {mockStories.map((story) => (
                <div
                  key={story.id}
                  className="bg-gradient-to-br from-purple-900/40 to-black/40 backdrop-blur-sm rounded-xl border border-purple-700/30 p-5 hover:border-purple-500/50 transition-all"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-700 rounded-full flex items-center justify-center">
                        <User className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white text-sm">{story.representative}</h3>
                        <div className="flex items-center space-x-1 text-xs text-gray-400">
                          <Clock className="w-3 h-3" />
                          <span>{story.timestamp}</span>
                        </div>
                      </div>
                    </div>
                    <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-xs font-medium border border-purple-500/30">
                      {story.category}
                    </span>
                  </div>

                  <h4 className="text-lg font-bold text-white mb-2">{story.title}</h4>
                  <p className="text-sm text-gray-300 mb-4 leading-relaxed">{story.summary}</p>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-medium text-gray-400">Alignment:</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${
                              story.alignment >= 80 ? 'bg-green-500' :
                              story.alignment >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${story.alignment}%` }}
                          />
                        </div>
                        <span className="text-xs font-bold text-white">{story.alignment}%</span>
                      </div>
                    </div>
                    <button className="text-purple-400 hover:text-purple-300 font-medium text-xs">
                      View Details →
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Load More */}
            <div className="text-center mt-6">
              <button className="px-5 py-2.5 bg-purple-900/30 border border-purple-700/30 text-purple-300 rounded-lg text-sm font-medium hover:bg-purple-800/40 transition-all">
                Load More Stories
              </button>
            </div>
          </div>
        </main>
      </div>

      {/* Chat Sidebar */}
      <div 
        className={`bg-gradient-to-b from-purple-950 to-black border-l border-purple-800/30 flex flex-col transition-all duration-300 ease-in-out ${
          isChatExpanded ? 'w-80' : 'w-12'
        }`}
      >
        {/* Toggle Button */}
        <button
          onClick={() => setIsChatExpanded(!isChatExpanded)}
          className="absolute right-0 top-20 bg-purple-600 hover:bg-purple-700 p-2 rounded-l-lg transition-colors z-10"
        >
          {isChatExpanded ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>

        {isChatExpanded && (
          <>
            {/* Chat Header */}
            <div className="px-4 py-3 border-b border-purple-800/30">
              <h3 className="font-semibold text-white text-sm">Ask Questions</h3>
              <p className="text-xs text-purple-300">Get insights on the data</p>
            </div>

            {/* Suggested Questions */}
            {chatMessages.length === 0 && (
              <div className="px-4 py-3 space-y-2">
                <p className="text-xs font-medium text-purple-400 mb-2">SUGGESTED</p>
                {suggestedQuestions.map((question, idx) => (
                  <button
                    key={idx}
                    onClick={() => setInputMessage(question)}
                    className="w-full text-left px-3 py-2 bg-purple-900/30 hover:bg-purple-800/40 rounded-lg text-xs text-purple-200 transition-all border border-purple-700/30"
                  >
                    {question}
                  </button>
                ))}
              </div>
            )}

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[85%] px-3 py-2 rounded-lg ${
                      msg.type === 'user'
                        ? 'bg-purple-600 text-white'
                        : 'bg-purple-900/50 text-purple-100 border border-purple-700/30'
                    }`}
                  >
                    <p className="text-xs leading-relaxed">{msg.text}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Chat Input */}
            <div className="px-4 py-3 border-t border-purple-800/30">
              <div className="flex space-
