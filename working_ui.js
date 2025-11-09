import React, { useState, useEffect } from 'react';
import { Send, User, FileText, Clock, ChevronRight, Sparkles, LogIn, LogOut } from 'lucide-react';

// --- Simple storage shim so this preview runs even if window.storage isn't provided ---
function ensureStorageShim() {
  if (typeof window === 'undefined') return;
  if (!window.storage) {
    const NS_PREFIX = 'ptui:';
    window.storage = {
      async get(key) {
        const v = localStorage.getItem(NS_PREFIX + key);
        return v ? { value: v } : { value: null };
      },
      async set(key, value) {
        localStorage.setItem(NS_PREFIX + key, value);
        return { ok: true };
      },
      async delete(key) {
        localStorage.removeItem(NS_PREFIX + key);
        return { ok: true };
      },
      async list(prefix) {
        const keys = [];
        for (let i = 0; i < localStorage.length; i++) {
          const k = localStorage.key(i);
          if (k && k.startsWith(NS_PREFIX + prefix)) {
            keys.push(k.replace(NS_PREFIX, ''));
          }
        }
        return { keys };
      }
    };
  }
}

const TOPIC_OPTIONS = [
  'Housing & Development',
  'Education Funding & Property Tax',
  'Taxes & Economic Policy',
  'Environment & Climate',
  'Workforce & Labor',
  'Healthcare & Mental Health',
  'Public Safety & Justice',
  'Infrastructure & Energy',
  'Civic & Electoral Reform',
];

export default function PoliticalTransparencyUI() {
  ensureStorageShim();

  const [chatMessages, setChatMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isChatExpanded, setIsChatExpanded] = useState(true);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [showLogin, setShowLogin] = useState(false);
  const [userData, setUserData] = useState(null);
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [signupName, setSignupName] = useState('');
  const [signupEmail, setSignupEmail] = useState('');
  const [signupPassword, setSignupPassword] = useState('');
  const [isSignup, setIsSignup] = useState(false);

  // Onboarding modal flags & fields
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [obFullName, setObFullName] = useState('');
  const [obTopics, setObTopics] = useState([]);
  const [obOtherTopics, setObOtherTopics] = useState('');
  const [obReadingLevel, setObReadingLevel] = useState(null); // 1 | 2 | 3
  const [obLocations, setObLocations] = useState('');

  useEffect(() => {
    loadUserData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadUserData = async () => {
    try {
      const sessionResult = await window.storage.get('current_session', false);
      if (sessionResult?.value) {
        const session = JSON.parse(sessionResult.value);
        const userResult = await window.storage.get(`user:${session.userId}`, false);
        if (userResult?.value) {
          const user = JSON.parse(userResult.value);
          setUserData(user);
          setIsLoggedIn(true);
          // Optionally show onboarding if no prefs exist; disabled by default
          // const prefs = await window.storage.get(`user_prefs:${user.id}`, false);
          // if (!prefs?.value) setShowOnboarding(true);
        }
      }
    } catch (error) {
      console.log('No active session');
    }
  };

  const handleLogin = async () => {
    try {
      const userListResult = await window.storage.list('user:', false);
      if (!userListResult?.keys?.length) {
        alert('No account found. Please sign up.');
        return;
      }
      for (const key of userListResult.keys) {
        try {
          const userResult = await window.storage.get(key, false);
          if (userResult?.value) {
            const user = JSON.parse(userResult.value);
            if (user.email === loginEmail && user.password === loginPassword) {
              setUserData(user);
              setIsLoggedIn(true);
              await window.storage.set('current_session', JSON.stringify({ userId: user.id }), false);
              setShowLogin(false);
              setLoginEmail('');
              setLoginPassword('');
              return;
            }
          }
        } catch (e) {
          continue;
        }
      }
      alert('Invalid email or password');
    } catch (error) {
      console.error('Login error:', error);
      alert('Login failed. Please try again.');
    }
  };

  const handleSignup = async () => {
    if (!signupName || !signupEmail || !signupPassword) {
      alert('Please fill in all fields');
      return;
    }

    try {
      const userId = `user_${Date.now()}`;
      const newUser = {
        id: userId,
        name: signupName,
        email: signupEmail,
        password: signupPassword,
        createdAt: new Date().toISOString(),
      };

      await window.storage.set(`user:${userId}`, JSON.stringify(newUser), false);
      await window.storage.set('current_session', JSON.stringify({ userId }), false);

      setUserData(newUser);
      setIsLoggedIn(true);
      setShowLogin(false);
      setSignupName('');
      setSignupEmail('');
      setSignupPassword('');
      setIsSignup(false);

      // Prefill onboarding and show
      setObFullName(newUser.name || '');
      setShowOnboarding(true);
    } catch (error) {
      console.error('Signup error:', error);
      alert('Signup failed. Please try again.');
    }
  };

  const handleLogout = async () => {
    try {
      await window.storage.delete('current_session', false);
      setIsLoggedIn(false);
      setUserData(null);
      setChatMessages([]);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const handleSendMessage = () => {
    if (inputMessage.trim()) {
      setChatMessages((prev) => [...prev, { type: 'user', text: inputMessage }]);
      setInputMessage('');
      setTimeout(() => {
        setChatMessages((prev) => [
          ...prev,
          { type: 'assistant', text: `I'm Relay, your political transparency assistant. How can I help you understand this better?` },
        ]);
      }, 500);
    }
  };

  const toggleTopic = (topic) => {
    setObTopics((prev) => (prev.includes(topic) ? prev.filter((t) => t !== topic) : [...prev, topic]));
  };

  const validateLocations = (val) => {
    const parts = val.split(';').map((s) => s.trim()).filter(Boolean);
    if (parts.length === 0) return false;
    for (const p of parts) {
      const [city, state] = p.split(',').map((x) => (x || '').trim());
      if (!city || !state) return false;
    }
    return true;
  };

  const buildPrefsTextFile = (prefs) => {
    const lines = [
      `User Preferences`,
      `================`,
      `User ID: ${prefs.userId}`,
      `Full Name: ${prefs.fullName}`,
      `Reading Level: ${prefs.readingLevel}`,
      `Topics: ${prefs.topics.join(', ') || 'None selected'}`,
      `Other Topics: ${prefs.otherTopics || '—'}`,
      `Locations: ${prefs.locations}`,
      `Saved At: ${new Date().toISOString()}`,
    ];
    return lines.join('\n');
  };

  const triggerDownload = (filename, text) => {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    URL.revokeObjectURL(url);
    a.remove();
  };

  const handleOnboardingSubmit = async () => {
    if (!obFullName.trim()) {
      alert('Please enter your full name.');
      return;
    }
    if (!obReadingLevel) {
      alert('Please select a reading level.');
      return;
    }
    if (!obLocations.trim() || !validateLocations(obLocations)) {
      alert('Please provide locations in the format "City, State"; multiple locations separated by semicolons.');
      return;
    }
    try {
      const userId = userData.id;
      const prefs = {
        userId,
        fullName: obFullName.trim(),
        topics: obTopics,
        otherTopics: obOtherTopics.trim(),
        readingLevel: obReadingLevel,
        locations: obLocations.trim(),
      };

      await window.storage.set(`user_prefs:${userId}`, JSON.stringify(prefs), false);

      if (obFullName.trim() && obFullName.trim() !== userData.name) {
        const updatedUser = { ...userData, name: obFullName.trim() };
        await window.storage.set(`user:${userId}`, JSON.stringify(updatedUser), false);
        setUserData(updatedUser);
      }

      const txt = buildPrefsTextFile(prefs);
      await window.storage.set(`file:user_prefs:${userId}.txt`, txt, false);
      triggerDownload(`user_prefs_${userId}.txt`, txt);

      setShowOnboarding(false);
    } catch (e) {
      console.error('Onboarding save error:', e);
      alert('Unable to save your preferences. Please try again.');
    }
  };

  const suggestedQuestions = [
    'What bills did they vote on?',
    'Compare promises vs actions',
    'How do I contact them?',
    'Show recent updates',
  ];

  const mockStories = [
    {
      id: 1,
      title: 'Infrastructure Bill Vote Analysis',
      representative: 'Sen. Jane Smith',
      alignment: 87,
      category: 'Economy',
      summary:
        'Voted in favor of infrastructure spending, consistent with campaign promises on job creation.',
      timestamp: '2 hours ago',
    },
    {
      id: 2,
      title: 'Healthcare Reform Statement',
      representative: 'Rep. John Doe',
      alignment: 65,
      category: 'Healthcare',
      summary:
        'Public statements support expansion, but recent committee votes show mixed record.',
      timestamp: '5 hours ago',
    },
    {
      id: 3,
      title: 'Climate Policy Update',
      representative: 'Sen. Maria Garcia',
      alignment: 92,
      category: 'Environment',
      summary:
        'Strong alignment between campaign promises and legislative actions on renewable energy.',
      timestamp: '1 day ago',
    },
    {
      id: 4,
      title: 'Education Funding Vote',
      representative: 'Rep. Michael Chen',
      alignment: 78,
      category: 'Education',
      summary:
        'Supported increased education funding, aligning with campaign trail commitments.',
      timestamp: '2 days ago',
    },
  ];

  return (
    <div className="flex h-screen bg-black text-white overflow-hidden">
      {/* User Profile Button */}
      <div className="fixed top-4 left-4 z-50">
        {isLoggedIn ? (
          <div className="flex items-center space-x-3 bg-white/5 backdrop-blur-xl px-4 py-2 rounded-full border border-white/20 shadow-lg">
            <div className="w-8 h-8 bg-white/10 rounded-full flex items-center justify-center font-bold text-sm text-white">
              {userData?.name?.charAt(0).toUpperCase()}
            </div>
            <span className="text-sm font-medium text-white">{userData?.name}</span>
            <button
              onClick={handleLogout}
              className="ml-2 p-1 hover:bg-white/10 rounded-full transition-colors"
            >
              <LogOut className="w-4 h-4 text-gray-200" />
            </button>
          </div>
        ) : (
          <button
            onClick={() => setShowLogin(true)}
            className="flex items-center space-x-2 bg-white/5 backdrop-blur-xl px-4 py-2 rounded-full border border-white/20 shadow-lg hover:bg-white/10 transition-all"
          >
            <LogIn className="w-4 h-4 text-gray-100" />
            <span className="text-sm font-medium text-white">Sign In</span>
          </button>
        )}
      </div>

      {/* Login Modal */}
      {showLogin && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gradient-to-br from-neutral-950 to-black border border-white/15 rounded-2xl p-6 max-w-md w-full shadow-2xl">
            <h2 className="text-2xl font-bold mb-6 text-white">
              {isSignup ? 'Create Account' : 'Welcome Back'}
            </h2>

            {isSignup && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">Name</label>
                <input
                  type="text"
                  value={signupName}
                  onChange={(e) => setSignupName(e.target.value)}
                  className="w-full px-4 py-2 bg-white/5 border border-white/15 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-white/30 focus:border-transparent"
                  placeholder="Your name"
                />
              </div>
            )}

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
              <input
                type="email"
                value={isSignup ? signupEmail : loginEmail}
                onChange={(e) => (isSignup ? setSignupEmail(e.target.value) : setLoginEmail(e.target.value))}
                className="w-full px-4 py-2 bg-white/5 border border-white/15 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-white/30 focus:border-transparent"
                placeholder="you@example.com"
              />
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">Password</label>
              <input
                type="password"
                value={isSignup ? signupPassword : loginPassword}
                onChange={(e) => (isSignup ? setSignupPassword(e.target.value) : setLoginPassword(e.target.value))}
                onKeyDown={(e) => e.key === 'Enter' && (isSignup ? handleSignup() : handleLogin())}
                className="w-full px-4 py-2 bg-white/5 border border-white/15 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-white/30 focus:border-transparent"
                placeholder="••••••••"
              />
            </div>

            <button
              onClick={isSignup ? handleSignup : handleLogin}
              className="w-full py-3 bg-white text-black rounded-lg font-semibold hover:bg-gray-200 transition-all shadow"
            >
              {isSignup ? 'Sign Up' : 'Sign In'}
            </button>

            <button
              onClick={() => setIsSignup(!isSignup)}
              className="w-full mt-3 text-sm text-gray-300 hover:text-white"
            >
              {isSignup ? 'Already have an account? Sign in' : "Don't have an account? Sign up"}
            </button>

            <button
              onClick={() => setShowLogin(false)}
              className="w-full mt-2 text-sm text-gray-500 hover:text-gray-300"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Onboarding Modal (post-signup) */}
      {showOnboarding && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-neutral-950 border border-white/15 rounded-2xl p-6 max-w-2xl w-full shadow-2xl overflow-y-auto max-h-[90vh]">
            <h2 className="text-2xl font-bold text-white mb-1">Tell us about you</h2>
            <p className="text-sm text-gray-400 mb-6">
              We’ll personalize updates based on your interests and location. This appears only once.
            </p>

            {/* Full Name */}
            <div className="mb-5">
              <label className="block text-sm font-medium text-gray-300 mb-2">Full Name</label>
              <input
                type="text"
                value={obFullName}
                onChange={(e) => setObFullName(e.target.value)}
                className="w-full px-4 py-2 bg-white/5 border border-white/15 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-white/30 focus:border-transparent"
                placeholder="First Last"
              />
            </div>

            {/* Topics (multi-select) */}
            <div className="mb-5">
              <label className="block text-sm font-medium text-gray-300 mb-2">Topics of Interest (select any)</label>
              <div className="flex flex-wrap gap-2">
                {TOPIC_OPTIONS.map((t) => {
                  const selected = obTopics.includes(t);
                  return (
                    <button
                      key={t}
                      type="button"
                      onClick={() => toggleTopic(t)}
                      className={`px-3 py-1.5 rounded-full text-sm border transition ${
                        selected ? 'bg-white text-black border-white' : 'bg-white/5 text-gray-200 border-white/15 hover:bg-white/10'
                      }`}
                    >
                      {t}
                    </button>
                  );
                })}
              </div>

              <div className="mt-3">
                <label className="block text-xs font-medium text-gray-400 mb-1">Other topics (optional)</label>
                <textarea
                  value={obOtherTopics}
                  onChange={(e) => setObOtherTopics(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 bg-white/5 border border-white/15 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-white/30 focus:border-transparent"
                  placeholder="Add any additional topics..."
                />
              </div>
            </div>

            {/* Reading Level (single-select) */}
            <div className="mb-5">
              <label className="block text-sm font-medium text-gray-300 mb-2">How in-depth would you like your writing to be?</label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                {[1, 2, 3].map((lvl) => {
                  const labels = {
                    1: 'Level 1: Clear & concise',
                    2: 'Level 2: Detailed & technical',
                    3: 'Level 3: Highly technical / policy-style',
                  };
                  const selected = obReadingLevel === lvl;
                  return (
                    <button
                      key={lvl}
                      type="button"
                      onClick={() => setObReadingLevel(lvl)}
                      className={`text-left p-3 rounded-lg border transition ${
                        selected ? 'bg-white text-black border-white' : 'bg-white/5 text-gray-200 border-white/15 hover:bg:white/10'
                      }`}
                    >
                      <div className="text-sm font-semibold">Level {lvl}</div>
                      <div className="text-xs text-gray-300 mt-1">{labels[lvl]}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Locations */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">What cities or states would you like insights on?</label>
              <p className="text-xs text-gray-400 mb-2">
                Format: <span className="font-mono">City, State</span>. Multiple locations separated by a semicolon.
                <br />
                Example: <span className="font-mono">Chicago, IL; Springfield, IL; Boston, MA</span>
              </p>
              <input
                type="text"
                value={obLocations}
                onChange={(e) => setObLocations(e.target.value)}
                className="w-full px-4 py-2 bg-white/5 border border-white/15 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-white/30 focus:border-transparent"
                placeholder="City, ST; City, ST; ..."
              />
            </div>

            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setShowOnboarding(false)}
                className="px-4 py-2 rounded-lg border border-white/15 text-gray-200 hover:bg-white/10 transition"
              >
                Skip for now
              </button>
              <button
                onClick={handleOnboardingSubmit}
                className="px-4 py-2 rounded-lg bg:white text-black font-semibold hover:bg-gray-200 transition"
                style={{ backgroundColor: 'white' }}
              >
                Save & Continue
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-gradient-to-r from-neutral-950 via-neutral-900 to-black border-b border-white/10 px-6 py-3.5 shadow-lg">
          <div className="flex items-center justify-center space-x-3">
            <div className="w-9 h-9 bg-white/10 rounded-xl flex items-center justify-center shadow">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Political Transparency</h1>
              <p className="text-xs text-gray-300">Track what they say vs. what they do</p>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto px-6 py-6 bg-gradient-to-br from-black via-neutral-950/10 to-black">
          <div className="max-w-4xl mx-auto">
            {/* Welcome Section */}
            <div className="mb-8">
              <div className="flex items-center space-x-3 mb-3">
                <Sparkles className="w-7 h-7 text-gray-300" />
                <h2 className="text-4xl font-bold text-white">
                  Hey {isLoggedIn && userData?.name ? userData.name : 'there'}!
                </h2>
              </div>
              <p className="text-lg text-gray-300 ml-10">Here are some interesting stories for you today</p>
            </div>

            {/* Filter Tabs */}
            <div className="flex space-x-2 mb-6 overflow-x-auto pb-2 scrollbar-hide">
              {['All', 'Economy', 'Healthcare', 'Environment', 'Education', 'Justice'].map((category) => (
                <button
                  key={category}
                  className={`px-4 py-2 rounded-xl text-sm font-semibold whitespace-nowrap transition-all ${
                    category === 'All'
                      ? 'bg-white text-black shadow'
                      : 'bg-white/5 text-gray-200 hover:bg-white/10 border border-white/15'
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>

            {/* News Feed */}
            <div className="space-y-4">
              {mockStories.map((story, idx) => (
                <div
                  key={story.id}
                  className="bg-gradient-to-br from-neutral-950/40 via-neutral-900/20 to-black/60 backdrop-blur-md rounded-2xl border border-white/10 p-5 hover:border-white/25 hover:shadow-xl transition-all cursor-pointer"
                  style={{ animationDelay: `${idx * 100}ms` }}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-11 h-11 bg-white/10 rounded-full flex items-center justify-center shadow">
                        <User className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">{story.representative}</h3>
                        <div className="flex items-center space-x-1.5 text-xs text-gray-400">
                          <Clock className="w-3 h-3" />
                          <span>{story.timestamp}</span>
                        </div>
                      </div>
                    </div>
                    <span className="px-3 py-1.5 bg-white/5 text-gray-200 rounded-full text-xs font-semibold border border-white/15">
                      {story.category}
                    </span>
                  </div>

                  <h4 className="text-lg font-bold text-white mb-2">{story.title}</h4>
                  <p className="text-sm text-gray-300 mb-4 leading-relaxed">{story.summary}</p>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className="text-xs font-medium text-gray-400">Alignment:</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-28 h-2 bg-white/10 rounded-full overflow-hidden backdrop-blur-sm">
                          <div className="h-full rounded-full bg-white" style={{ width: `${story.alignment}%` }} />
                        </div>
                        <span className="text-sm font-bold text-white">{story.alignment}%</span>
                      </div>
                    </div>
                    <button className="text-gray-200 hover:text-white font-semibold text-sm transition-colors">
                      View Details →
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Load More */}
            <div className="text-center mt-8">
              <button className="px-6 py-3 bg-white/5 border border-white/15 text-gray-100 rounded-xl font-semibold hover:bg-white/10 transition-all">
                Load More Stories
              </button>
            </div>
          </div>
        </main>
      </div>

      {/* Chat Sidebar - Relay */}
      <div
        className={`bg-gradient-to-b from-neutral-950/95 via-black to-black/95 backdrop-blur-xl border-l border-white/10 flex flex-col transition-all duration-300 ease-in-out shadow-2xl ${
          isChatExpanded ? 'w-80' : 'w-0'
        }`}
      >
        {isChatExpanded && (
          <>
            {/* Chat Header */}
            <div className="px-5 py-4 border-b border-white/10 bg-gradient-to-r from-neutral-900/50 to-transparent">
              <div className="flex items-center space-x-2 mb-1">
                <Sparkles className="w-5 h-5 text-gray-200" />
                <h3 className="font-bold text-white">Relay</h3>
              </div>
              <p className="text-xs text-gray-400">Your AI transparency assistant</p>
            </div>

            {/* Suggested Questions */}
            {chatMessages.length === 0 && (
              <div className="px-5 py-4 space-y-2">
                <p className="text-xs font-semibold text-gray-400 mb-3">SUGGESTED QUESTIONS</p>
                {suggestedQuestions.map((question, idx) => (
                  <button
                    key={idx}
                    onClick={() => setInputMessage(question)}
                    className="w-full text-left px-3 py-2.5 bg-white/5 hover:bg-white/10 rounded-xl text-xs text-gray-100 transition-all border border-white/15"
                  >
                    {question}
                  </button>
                ))}
              </div>
            )}

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
              {chatMessages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div
                    className={`max-w-[85%] px-3 py-2.5 rounded-xl ${
                      msg.type === 'user' ? 'bg-white text-black shadow' : 'bg-white/5 text-gray-100 border border-white/15'
                    }`}
                  >
                    <p className="text-xs leading-relaxed">{msg.text}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Chat Input */}
            <div className="px-5 py-4 border-t border-white/10 bg-gradient-to-r from-transparent to-neutral-900/30">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Ask Relay anything..."
                  className="flex-1 px-4 py-2.5 bg-white/5 border border-white/15 rounded-xl focus:ring-2 focus:ring-white/30 focus:border-transparent text-white placeholder-gray-400 text-sm"
                />
                <button onClick={handleSendMessage} className="p-2.5 bg-white text-black rounded-xl hover:bg-gray-200 transition-all shadow">
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Toggle Chat Button */}
      <button
        onClick={() => setIsChatExpanded(!isChatExpanded)}
        className="fixed right-4 bottom-6 bg-white text-black hover:bg-gray-200 p-4 rounded-full shadow-2xl transition-all z-40 hover:scale-110"
      >
        {isChatExpanded ? <ChevronRight className="w-5 h-5" /> : <Sparkles className="w-5 h-5" />}
      </button>
    </div>
  );
}
