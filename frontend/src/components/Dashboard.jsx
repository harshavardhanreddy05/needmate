import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { toast } from 'sonner';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Mic, Search, LogOut, History, Sparkles, ShoppingBag, Star, ExternalLink } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Slider } from './ui/slider';
import { Checkbox } from './ui/checkbox';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;


export default function Dashboard({ user, onLogout }) {
  const [query, setQuery] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [history, setHistory] = useState([]);
  const recognitionRef = useRef(null);

  // Filter sidebar state
  const [budgetRange, setBudgetRange] = useState([0, 10000]);
  const [selectedBrands, setSelectedBrands] = useState([]);
  const [selectedRatings, setSelectedRatings] = useState([]);
  const [bundleMaxPrices, setBundleMaxPrices] = useState({}); 
 const applyFilters = async () => {
  setLoading(true);
  try {
    const token = localStorage.getItem('token');
    const filters = {
      min_price: budgetRange[0],
      max_price: budgetRange[1],
      min_rating: selectedRatings.length > 0 ? Math.min(...selectedRatings) : undefined,
      category: selectedBrands.length > 0 ? selectedBrands[0] : undefined,
    };
    const response = await axios.post(
      `${API}/search`,
      {
        query,
        query_type: isListening ? 'voice' : 'text',
        filters,
      },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    setResults(response.data);
    fetchHistory();
    toast.success('Filters applied successfully!');
  } catch (error) {
    toast.error(error.response?.data?.detail || 'Failed to apply filters');
  } finally {
    setLoading(false);
  }
};


  useEffect(() => {
    // Initialize Web Speech API
    const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
    
    if (!SpeechRecognition) {
      console.warn('Speech Recognition API not supported in this browser');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      console.log('Voice recognition started');
    };

    recognition.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' ';
        } else {
          interimTranscript += transcript;
        }
      }

      // Update query with final transcript
      if (finalTranscript) {
        setQuery(prev => (prev + ' ' + finalTranscript).trim());
      }
    };

    recognition.onerror = (event) => {
      setIsListening(false);
      const errorMessages = {
        'no-speech': 'No speech detected. Please try again.',
        'audio-capture': 'No microphone found. Please check your microphone settings.',
        'network': 'Network error. Please check your internet connection.',
        'permission-denied': 'Microphone permission denied. Please allow microphone access in browser settings.',
        'not-allowed': 'Microphone access not allowed. Please enable microphone permissions.',
      };
      
      const errorMessage = errorMessages[event.error] || `Voice recognition error: ${event.error}`;
      console.error('Speech Recognition Error:', event.error);
      toast.error(errorMessage);
    };

    recognition.onend = () => {
      setIsListening(false);
      console.log('Voice recognition ended');
    };

    recognitionRef.current = recognition;

    // Fetch search history
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`${API}/history`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setHistory(response.data);
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  const startVoiceRecognition = () => {
    const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
    
    if (!SpeechRecognition) {
      toast.error('Voice recognition not supported in your browser. Try Chrome, Edge, or Safari.');
      return;
    }

    if (!recognitionRef.current) {
      toast.error('Voice recognition not initialized. Please refresh the page.');
      return;
    }

    // Check for microphone permissions (works in modern browsers)
    navigator.permissions
      .query({ name: 'microphone' })
      .then((result) => {
        if (result.state === 'denied') {
          toast.error('Microphone permission denied. Enable it in browser settings.');
          return;
        }
        
        try {
          setIsListening(true);
          recognitionRef.current.start();
          toast.success('Listening... speak now');
        } catch (error) {
          setIsListening(false);
          console.error('Error starting recognition:', error);
          toast.error('Failed to start voice recognition. Try again.');
        }
      })
      .catch((error) => {
        // Fallback for browsers that don't support permissions API
        console.log('Permissions API not supported, attempting voice recognition anyway');
        try {
          setIsListening(true);
          recognitionRef.current.start();
          toast.success('Listening... speak now');
        } catch (error) {
          setIsListening(false);
          console.error('Error starting recognition:', error);
          toast.error('Failed to start voice recognition. Try again.');
        }
      });
  };

  // const handleSearch = async (e) => {
  //   e.preventDefault();
  //   if (!query.trim()) {
  //     toast.error('Please enter a search query');
  //     return;
  //   }

  //   setLoading(true);
  //   try {
  //     const token = localStorage.getItem('token');
  //     const response = await axios.post(
  //       `${API}/search`,
  //       {
  //         query,
  //         query_type: isListening ? 'voice' : 'text'
  //       },
  //       {
  //         headers: { Authorization: `Bearer ${token}` }
  //       }
  //     );

  //     setResults(response.data);
  //     fetchHistory();
  //     toast.success(`Found products in ${response.data.category}`);
  //   } catch (error) {
  //     toast.error(error.response?.data?.detail || 'Search failed');
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  const handleSearch = async (e) => {
  e.preventDefault();
  if (!query.trim()) {
    toast.error('Please enter a search query');
    return;
  }

  setLoading(true);
  try {
    const token = localStorage.getItem('token');
    const response = await axios.post(
      `${API}/search`,
      {
        query,
        query_type: isListening ? 'voice' : 'text',
        filters: {
          price_min: budgetRange[0],
          price_max: budgetRange[1],
          ratings: selectedRatings,
        },
      },
      {
        headers: { Authorization: `Bearer ${token}` },
      }
    );

    setResults(response.data);
    fetchHistory();
    toast.success(`Found products in ${response.data.category}`);
  } catch (error) {
    toast.error(error.response?.data?.detail || 'Search failed');
  } finally {
    setLoading(false);
  }
};


  return (
    <div className="min-h-screen gradient-ocean">
      {/* Header */}
      <header className="glass border-b border-white/30">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900">NeedMate</h1>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600 hidden sm:inline">{user.email}</span>
            <Button
              onClick={onLogout}
              data-testid="logout-button"
              variant="outline"
              className="flex items-center gap-2 bg-white/80 hover:bg-white border-gray-200"
            >
              <LogOut className="w-4 h-4" />
              <span className="hidden sm:inline">Logout</span>
            </Button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Search Section */}
        <div className="mb-12 animate-fade-in">
          <div className="text-center mb-8">
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
              What are you looking for?
            </h2>
            <p className="text-base sm:text-lg text-gray-600">
              Describe your needs in natural language or use voice
            </p>
          </div>

          <form onSubmit={handleSearch} className="max-w-3xl mx-auto">
            <div className="glass rounded-2xl p-3 flex gap-3 shadow-xl">
              <Input
                type="text"
                data-testid="search-input"
                placeholder="E.g., 'I need a comfortable chair for back pain' or 'laptop for gaming'"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="flex-1 h-14 border-0 bg-transparent text-base focus-visible:ring-0 focus-visible:ring-offset-0"
              />
              <Button
                type="button"
                data-testid="voice-button"
                onClick={startVoiceRecognition}
                disabled={isListening}
                className={`h-14 w-14 rounded-xl ${
                  isListening
                    ? 'bg-red-500 hover:bg-red-600 animate-pulse'
                    : 'bg-gradient-to-br from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600'
                } text-white`}
              >
                <Mic className="w-5 h-5" />
              </Button>
              <Button
                type="submit"
                data-testid="search-button"
                disabled={loading}
                className="h-14 px-8 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-semibold rounded-xl btn-hover"
              >
                {loading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Searching...</span>
                  </div>
                ) : (
                  <>
                    <Search className="w-5 h-5" />
                  </>
                )}
              </Button>
            </div>
          </form>
        </div>
{/* Search and Filters Section */}
{results ? (
  <div className="flex gap-8">
    {/* Filter Sidebar */}
   





    {/* Main Content */}
    <div className="flex-1">
      <Tabs defaultValue="results" className="animate-slide-up">
        <TabsList className="glass mb-6 p-1 border border-white/30 flex">
          <TabsTrigger
            value="results"
            data-testid="results-tab"
            className="data-[state=active]:bg-white data-[state=active]:shadow-md flex-1"
          >
            <ShoppingBag className="w-4 h-4 mr-2" />
            Products
          </TabsTrigger>
          <TabsTrigger
            value="bundles"
            data-testid="bundles-tab"
            className="data-[state=active]:bg-white data-[state=active]:shadow-md flex-1"
          >
            <Sparkles className="w-4 h-4 mr-2 text-yellow-500" />
            Smart Bundling
          </TabsTrigger>
          <TabsTrigger
            value="history"
            data-testid="history-tab"
            className="data-[state=active]:bg-white data-[state=active]:shadow-md flex-1"
          >
            <History className="w-4 h-4 mr-2" />
            Search History
          </TabsTrigger>
        </TabsList>

        {/* Results Tab */}
        <TabsContent value="results" data-testid="results-content">
          {results.products && results.products.length > 0 ? (
            <div className="flex flex-col lg:flex-row gap-6">
              {/* Filter Sidebar - Visible only on large screens */}
             

              {/* Mobile Filter Button */}
              <div className="lg:hidden mb-4">
                <Button
                  onClick={() =>
                    document.getElementById('mobileFilters').classList.toggle('hidden')
                  }
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold"
                >
                  Show Filters
                </Button>

                {/* Collapsible Filters for Mobile */}
                <div
                  id="mobileFilters"
                  className="hidden bg-white p-5 shadow-lg rounded-2xl mt-4 space-y-4"
                >
                  <h2 className="text-lg font-bold text-gray-800">Filters</h2>

                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Price Range</h3>
                    <div className="flex gap-2">
                      <Input
                        type="number"
                        min={0}
                        value={budgetRange[0]}
                        onChange={e => {
                          const min = Number(e.target.value);
                          setBudgetRange([min, budgetRange[1]]);
                        }}
                        placeholder="Min"
                        className="w-1/2"
                      />
                      <Input
                        type="number"
                        min={0}
                        value={budgetRange[1]}
                        onChange={e => {
                          const max = Number(e.target.value);
                          setBudgetRange([budgetRange[0], max]);
                        }}
                        placeholder="Max"
                        className="w-1/2"
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">Enter min and max price</p>
                  </div>

                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Ratings</h3>
                    <div className="flex flex-col space-y-2">
                      {[4, 3, 2].map((rating) => (
                        <label
                          key={rating}
                          className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer"
                        >
                          <input
                            type="checkbox"
                            checked={selectedRatings.includes(rating)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedRatings([...selectedRatings, rating]);
                              } else {
                                setSelectedRatings(selectedRatings.filter((r) => r !== rating));
                              }
                            }}
                            className="accent-yellow-500"
                          />
                          {rating}+ Stars
                        </label>
                      ))}
                    </div>
                  </div>

                  <Button
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold"
                    onClick={applyFilters}
                  >
                    Apply Filters
                  </Button>
                </div>
              </div>

              {/* Products Grid */}
              <div className="flex-1 overflow-y-auto">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {results.products?.map((product) => (
                    <div
                      key={product.product_id}
                      className="glass rounded-2xl overflow-hidden shadow-lg hover:shadow-xl transition-shadow"
                    >
                      <div className="aspect-square bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center overflow-hidden">
                        {product.product_photo ? (
                          <img
                            src={product.product_photo}
                            alt={product.product_title}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <ShoppingBag className="w-16 h-16 text-gray-400" />
                        )}
                      </div>
                      <div className="p-4">
                        <h4 className="font-semibold text-gray-900 mb-2 line-clamp-2 text-sm">
                          {product.product_title}
                        </h4>
                        <div className="flex items-center justify-between mb-3">
                          {product.product_price && (
                            <span className="text-lg font-bold text-blue-600">
                              {product.product_price}
                            </span>
                          )}
                          {product.product_rating && (
                            <div className="flex items-center gap-1">
                              <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />
                              <span className="text-sm text-gray-700">
                                {Number(product.product_rating).toFixed(1)}
                              </span>
                            </div>
                          )}
                        </div>
                        <a
                          href={product.product_url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <Button className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white">
                            Buy Now
                            <ExternalLink className="w-4 h-4 ml-2" />
                          </Button>
                        </a>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="glass rounded-2xl p-12 text-center">
              <ShoppingBag className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No products found. Try a different search.</p>
            </div>
          )}
        </TabsContent>

        {/* Smart Bundling Tab *
        {/* <TabsContent value="bundles">
  {results?.bundles && results.bundles.length > 0 ? (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
      {results.bundles.map((bundle, index) => (
        <div
          key={index}
          className="glass rounded-2xl overflow-hidden shadow-lg p-4"
        >
          <h4 className="font-bold text-gray-900 mb-2">{bundle.title}</h4>
          <p className="text-gray-600 mb-3">{bundle.description}</p>
          <div className="flex flex-col gap-2">
            {bundle.products.map((product) => (
              <a
                key={product.product_id}
                href={product.product_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                {product.product_title} - {product.product_price}
              </a>
            ))}
          </div>
        </div>
      ))}
    </div>
  ) : (
    <div className="glass rounded-2xl p-12 text-center">
      <Sparkles className="w-10 h-10 text-yellow-500 mx-auto mb-4" />
      <p className="text-gray-600">No smart bundles available yet. Perform a search above to generate bundles.</p>
    </div>
  )}
</TabsContent> */}
<TabsContent value="bundles">
  {results?.bundles && results.bundles.length > 0 ? (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
      {results.bundles.map((bundle, index) => {
        const bundleMaxPrice = bundleMaxPrices[index] || '';

        // Filter products based on max price
        const filteredProducts = bundle.products.filter((product) => {
          if (!bundleMaxPrice) return true;
          const priceNumber = Number(
            String(product.product_price).replace(/[^0-9.]/g, '')
          );
          return priceNumber <= Number(bundleMaxPrice);
        });

        return (
          <div
            key={index}
            className="glass rounded-2xl overflow-hidden shadow-lg p-4 w-full max-w-sm mx-auto"
          >
            <h4 className="font-bold text-gray-900 mb-2">{bundle.title}</h4>
            {bundle.description && (
              <p className="text-gray-600 mb-3">{bundle.description}</p>
            )}

            {/* Max Price Input */}
            <div className="mb-4 flex items-center gap-3 w-full max-w-xs">
              <label className="text-sm font-medium text-gray-700">Max Price:</label>
              <Input
                type="number"
                min={0}
                placeholder="e.g., 5000"
                value={bundleMaxPrice}
                onChange={(e) =>
                  setBundleMaxPrices({
                    ...bundleMaxPrices,
                    [index]: e.target.value,
                  })
                }
                className="flex-1"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {filteredProducts.map((product) => (
                <div
                  key={product.product_id}
                  className="glass rounded-xl overflow-hidden shadow hover:shadow-md transition-shadow flex flex-col"
                >
                  <div className="aspect-square bg-gray-100 flex items-center justify-center overflow-hidden">
                    {product.product_photo ? (
                      <img
                        src={product.product_photo}
                        alt={product.product_title}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <ShoppingBag className="w-12 h-12 text-gray-400" />
                    )}
                  </div>
                  <div className="p-3 flex-1 flex flex-col justify-between">
                    <div className="mb-2">
                      <h5 className="text-sm font-medium text-gray-900 line-clamp-2">
                        {product.product_title}
                      </h5>
                      {product.product_price && (
                        <p className="text-blue-600 font-bold">{product.product_price}</p>
                      )}
                    </div>
                    <Button
                      size="sm"
                      onClick={() =>
                        window.open(product.product_url, '_blank', 'noopener')
                      }
                      className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white mt-auto"
                    >
                      Buy
                      <ExternalLink className="w-4 h-4 ml-1" />
                    </Button>
                  </div>
                </div>
              ))}
              {filteredProducts.length === 0 && (
                <p className="text-gray-500 col-span-full text-center">
                  No products under the selected max price.
                </p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  ) : (
    <div className="glass rounded-2xl p-12 text-center">
      <Sparkles className="w-10 h-10 text-yellow-500 mx-auto mb-4" />
      <p className="text-gray-600">
        No smart bundles available yet. Perform a search above to generate bundles.
      </p>
    </div>
  )}
</TabsContent>




        {/* History Tab */}
        <TabsContent value="history">
          {history.length > 0 ? (
            <div className="space-y-3">
              {history.map((item, index) => (
                <div
                  key={item.id}
                  className="glass rounded-xl p-4 flex items-center justify-between hover:shadow-lg transition-shadow"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-100 to-cyan-100 rounded-lg flex items-center justify-center">
                      {item.query_type === 'voice' ? (
                        <Mic className="w-5 h-5 text-blue-600" />
                      ) : (
                        <Search className="w-5 h-5 text-cyan-600" />
                      )}
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{item.query}</p>
                      <p className="text-sm text-gray-500">
                        {new Date(item.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="glass rounded-2xl p-12 text-center">
              <History className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No search history yet</p>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  </div>
) : (
  // Default empty state (no results yet)
  <div className="glass rounded-2xl p-12 text-center mt-12">
    <Search className="w-16 h-16 text-gray-400 mx-auto mb-4" />
    <p className="text-gray-600">Start searching to see products here</p>
  </div>
)}
      </div>
    </div>
  );
}