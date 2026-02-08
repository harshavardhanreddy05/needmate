# # from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
# # from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# # from dotenv import load_dotenv
# # from starlette.middleware.cors import CORSMiddleware
# # from motor.motor_asyncio import AsyncIOMotorClient
# # import os
# # import logging
# # from pathlib import Path
# # from pydantic import BaseModel, Field, ConfigDict, EmailStr
# # from typing import List, Optional
# # import uuid
# # from datetime import datetime, timezone, timedelta
# # from passlib.context import CryptContext
# # import jwt
# # import httpx
# # import re

# # ROOT_DIR = Path(__file__).parent
# # load_dotenv(ROOT_DIR / '.env')

# # # MongoDB connection
# # mongo_url = os.environ['MONGO_URL']
# # client = AsyncIOMotorClient(mongo_url)
# # db = client[os.environ['DB_NAME']]

# # # Password hashing
# # pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # # JWT settings
# # JWT_SECRET = os.environ.get('JWT_SECRET_KEY', 'expense-tracker-secret-key-2024')
# # JWT_ALGORITHM = "HS256"
# # ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# # # Security
# # security = HTTPBearer()

# # # Fast keyword-based classification (replaces heavy ML model)
# # CATEGORY_KEYWORDS = {
# #     "electronics": ["electronics", "electronic", "gadget", "device", "tech", "technology"],
# #     "computers": ["computer", "laptop", "pc", "desktop", "monitor", "keyboard", "mouse", "gaming"],
# #     "phones": ["phone", "smartphone", "mobile", "iphone", "android", "samsung", "tablet"],
# #     "clothing": ["clothing", "clothes", "shirt", "pants", "dress", "jacket", "wear", "fashion"],
# #     "shoes": ["shoe", "shoes", "sneaker", "boot", "sandal", "footwear"],
# #     "books": ["book", "books", "novel", "reading", "literature", "textbook", "ebook"],
# #     "home": ["home", "house", "decor", "decoration", "living", "bedroom", "interior"],
# #     "kitchen": ["kitchen", "cooking", "cookware", "utensil", "appliance", "pot", "pan"],
# #     "furniture": ["furniture", "chair", "table", "desk", "sofa", "couch", "bed", "cabinet"],
# #     "toys": ["toy", "toys", "game", "play", "kids", "children", "puzzle"],
# #     "sports": ["sport", "sports", "fitness", "exercise", "gym", "athletic", "workout"],
# #     "health": ["health", "medical", "wellness", "care", "medicine", "pain", "therapy"],
# #     "beauty": ["beauty", "cosmetic", "makeup", "skincare", "hair", "perfume"],
# #     "jewelry": ["jewelry", "jewellery", "ring", "necklace", "bracelet", "watch"],
# #     "automotive": ["car", "automotive", "vehicle", "auto", "motorcycle", "bike"]
# # }

# # def classify_query(query: str):
# #     """Fast keyword-based classification"""
# #     query_lower = query.lower()
# #     scores = {}
    
# #     for category, keywords in CATEGORY_KEYWORDS.items():
# #         score = 0
# #         for keyword in keywords:
# #             if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
# #                 score += 1
# #         scores[category] = score
    
# #     # Get category with highest score
# #     if max(scores.values()) > 0:
# #         best_category = max(scores.items(), key=lambda x: x[1])[0]
# #         confidence = min(scores[best_category] * 0.3, 0.95)  # Scale to confidence
# #     else:
# #         # Default to general search
# #         best_category = "electronics"
# #         confidence = 0.5
    
# #     return best_category, confidence

# # # RapidAPI settings
# # RAPID_API_KEY = "3227ea0400mshbc7f9fb81508577p189076jsnf5b6c369e941"
# # RAPID_API_HOST = "real-time-product-search.p.rapidapi.com"

# # # Create the main app without a prefix
# # app = FastAPI()

# # # Create a router with the /api prefix
# # api_router = APIRouter(prefix="/api")

# # # Models
# # class User(BaseModel):
# #     model_config = ConfigDict(extra="ignore")
# #     id: str = Field(default_factory=lambda: str(uuid.uuid4()))
# #     email: EmailStr
# #     hashed_password: str
# #     created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# # class UserRegister(BaseModel):
# #     email: EmailStr
# #     password: str

# # class UserLogin(BaseModel):
# #     email: EmailStr
# #     password: str

# # class Token(BaseModel):
# #     access_token: str
# #     token_type: str
# #     user: dict

# # class FilterOptions(BaseModel):
# #     min_price: Optional[float] = None
# #     max_price: Optional[float] = None
# #     min_rating: Optional[float] = None
# #     category: Optional[str] = None  # To filter by category


# # class SearchHistory(BaseModel):
# #     model_config = ConfigDict(extra="ignore")
# #     id: str = Field(default_factory=lambda: str(uuid.uuid4()))
# #     user_id: str
# #     query: str
# #     query_type: str  # 'text' or 'voice'
# #     timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# # # class SearchRequest(BaseModel):
# # #     query: str
# # #     query_type: str = 'text'
# # class SearchRequest(BaseModel):
# #     query: str
# #     query_type: str = 'text'
# #     filters: Optional[FilterOptions] = None

# # class ProductResult(BaseModel):
# #     product_id: str
# #     product_title: str
# #     product_photo: Optional[str]
# #     product_price: Optional[str]
# #     product_rating: Optional[float]
# #     product_url: str

# # class SearchResponse(BaseModel):
# #     category: str
# #     confidence: float
# #     products: List[ProductResult]

# # # appliying filters

# # def apply_filters(products: List[ProductResult], filters: Optional[FilterOptions]):
# #     if not filters:
# #         return products

# #     filtered = []

# #     for product in products:
# #         # Price filtering
# #         price = None
# #         if product.product_price:
# #             try:
# #                 price = float(re.sub(r'[^\d.]', '', str(product.product_price)))
# #             except:
# #                 price = None

# #         if filters.min_price and (price is None or price < filters.min_price):
# #             continue
# #         if filters.max_price and (price is None or price > filters.max_price):
# #             continue

# #         # Rating filtering
# #         if filters.min_rating and (product.product_rating is None or product.product_rating < filters.min_rating):
# #             continue

# #         # Category filtering
# #         if filters.category and filters.category.lower() not in product.product_title.lower():
# #             continue

# #         filtered.append(product)

# #     return filtered

# # # Helper functions
# # def hash_password(password: str) -> str:
# #     return pwd_context.hash(password)

# # def verify_password(plain_password: str, hashed_password: str) -> bool:
# #     return pwd_context.verify(plain_password, hashed_password)

# # def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
# #     to_encode = data.copy()
# #     if expires_delta:
# #         expire = datetime.now(timezone.utc) + expires_delta
# #     else:
# #         expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
# #     to_encode.update({"exp": expire})
# #     encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
# #     return encoded_jwt

# # async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
# #     try:
# #         token = credentials.credentials
# #         payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
# #         user_id: str = payload.get("sub")
# #         if user_id is None:
# #             raise HTTPException(status_code=401, detail="Invalid authentication credentials")
# #         user = await db.users.find_one({"id": user_id}, {"_id": 0})
# #         if user is None:
# #             raise HTTPException(status_code=401, detail="User not found")
# #         return user
# #     except jwt.ExpiredSignatureError:
# #         raise HTTPException(status_code=401, detail="Token has expired")
# #     except jwt.PyJWTError:
# #         raise HTTPException(status_code=401, detail="Could not validate credentials")

# # # Routes
# # @api_router.get("/")
# # async def root():
# #     return {"message": "NeedMate API is running"}

# # @api_router.post("/auth/register", response_model=Token)
# # async def register(user_data: UserRegister):
# #     # Check if user exists
# #     existing_user = await db.users.find_one({"email": user_data.email})
# #     if existing_user:
# #         raise HTTPException(status_code=400, detail="Email already registered")
    
# #     # Create new user
# #     user = User(
# #         email=user_data.email,
# #         hashed_password=hash_password(user_data.password)
# #     )
    
# #     user_dict = user.model_dump()
# #     user_dict['created_at'] = user_dict['created_at'].isoformat()
    
# #     await db.users.insert_one(user_dict)
    
# #     # Create access token
# #     access_token = create_access_token(data={"sub": user.id})
    
# #     return Token(
# #         access_token=access_token,
# #         token_type="bearer",
# #         user={"id": user.id, "email": user.email}
# #     )

# # @api_router.post("/auth/login", response_model=Token)
# # async def login(user_data: UserLogin):
# #     # Find user
# #     user = await db.users.find_one({"email": user_data.email}, {"_id": 0})
# #     if not user:
# #         raise HTTPException(status_code=401, detail="Invalid email or password")
    
# #     # Verify password
# #     if not verify_password(user_data.password, user["hashed_password"]):
# #         raise HTTPException(status_code=401, detail="Invalid email or password")
    
# #     # Create access token
# #     access_token = create_access_token(data={"sub": user["id"]})
    
# #     return Token(
# #         access_token=access_token,
# #         token_type="bearer",
# #         user={"id": user["id"], "email": user["email"]}
# #     )

# # @api_router.get("/auth/me")
# # async def get_me(current_user: dict = Depends(get_current_user)):
# #     return {"id": current_user["id"], "email": current_user["email"]}

# # # @api_router.post("/search", response_model=SearchResponse)
# # # async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
# # #     try:
# # #         # Fast keyword-based classification
# # #         category, confidence = classify_query(search_data.query)
        
# # #         # Search products using RapidAPI
# # #         async with httpx.AsyncClient() as client:
# # #             headers = {
# # #                 "X-RapidAPI-Key": RAPID_API_KEY,
# # #                 "X-RapidAPI-Host": RAPID_API_HOST
# # #             }
            
# # #             params = {
# # #                 "q": search_data.query,
# # #                 "country": "in",
# # #                 "language": "en"
# # #             }
            
# # #             response = await client.get(
# # #                 f"https://{RAPID_API_HOST}/search-v2",
# # #                 headers=headers,
# # #                 params=params,
# # #                 timeout=15.0
# # #             )
            
# # #             if response.status_code == 200:
# # #                 data = response.json()
# # #                 products = []
                
# # #                 # Parse products from response
# # #                 if data.get('status') == 'OK':
# # #                     response_data = data.get('data', {})
# # #                     # Get products from the nested structure
# # #                     product_data = response_data.get('products', []) if isinstance(response_data, dict) else []
                    
# # #                     # Limit to 12 products
# # #                     for item in (product_data if isinstance(product_data, list) else [])[:12]:
# # #                         # Extract price
# # #                         price = None
# # #                         if isinstance(item.get('offer'), dict):
# # #                             price = item['offer'].get('price')
# # #                         elif item.get('typical_price_range'):
# # #                             price = item.get('typical_price_range')
                        
# # #                         products.append(ProductResult(
# # #                             product_id=item.get('product_id', str(uuid.uuid4())),
# # #                             product_title=item.get('product_title', 'No title'),
# # #                             product_photo=item.get('product_photos', [None])[0] if item.get('product_photos') else item.get('product_photo'),
# # #                             product_price=price,
# # #                             product_rating=item.get('product_rating'),
# # #                             product_url=item.get('product_page_url', '#')
# # #                         ))
# # #                 else:
# # #                     products = []
# # #             else:
# # #                 products = []
        
# # #         # Save search history
# # #         history = SearchHistory(
# # #             user_id=current_user["id"],
# # #             query=search_data.query,
# # #             query_type=search_data.query_type
# # #         )
        
# # #         history_dict = history.model_dump()
# # #         history_dict['timestamp'] = history_dict['timestamp'].isoformat()
# # #         await db.search_history.insert_one(history_dict)
        
# # #         return SearchResponse(
# # #             category=category,
# # #             confidence=confidence,
# # #             products=products
# # #         )
    
# # #     except Exception as e:
# # #         logging.error(f"Search error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# # # @api_router.post("/search", response_model=SearchResponse)
# # # async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
# # #     try:
# # #         # Fast keyword-based classification
# # #         category, confidence = classify_query(search_data.query)

# # #         # Helper function to extract direct URLs
# # #         from urllib.parse import urlparse, parse_qs, unquote

# # #         def extract_direct_url(google_url: str):
# # #             """Extracts the actual product link if it's wrapped in a Google redirect"""
# # #             if not google_url:
# # #                 return "#"
# # #             if "google.com" in google_url and "url=" in google_url:
# # #                 parsed = urlparse(google_url)
# # #                 query = parse_qs(parsed.query)
# # #                 if "url" in query:
# # #                     return unquote(query["url"][0])
# # #             return google_url

# # #         # Search products using RapidAPI
# # #         async with httpx.AsyncClient() as client:
# # #             headers = {
# # #                 "X-RapidAPI-Key": RAPID_API_KEY,
# # #                 "X-RapidAPI-Host": RAPID_API_HOST
# # #             }
# # #             params = {
# # #                 "q": search_data.query,
# # #                 "country": "in",
# # #                 "language": "en"
# # #             }

# # #             # Log request details
# # #             logging.info(f"API Request URL: https://{RAPID_API_HOST}/search-v2")
# # #             logging.info(f"Search Query: {params['q']}")
            
# # #             response = await client.get(
# # #                 f"https://{RAPID_API_HOST}/search-v2",
# # #                 headers=headers,
# # #                 params=params,
# # #                 timeout=15.0
# # #             )
            
# # #             logging.info(f"API Status Code: {response.status_code}")

# # #             products = []

# # #             if response.status_code == 200:
# # #                 data = response.json()
# # #                 # Print raw API response
# # #                 logging.info("Raw API Response:")
# # #                 logging.info(f"{data}")
                
# # #                 if data.get('status') == 'OK':
# # #                     response_data = data.get('data', {})
# # #                     product_data = response_data.get('products', [])

# # #                     for item in (product_data if isinstance(product_data, list) else [])[:12]:
# # #                         # ✅ Extract price safely
# # #                         price = None
# # #                         if isinstance(item.get('offer'), dict):
# # #                             price = item['offer'].get('price')
# # #                         elif item.get('typical_price_range'):
# # #                             price = item.get('typical_price_range')

# # #                         # ✅ Find the most direct product link available
# # #                         direct_url = (
# # #                             item.get('product_link') or
# # #                             item.get('offer', {}).get('link') or
# # #                             item.get('product_page_url') or
# # #                             item.get('product_url') or
# # #                             "#"
# # #                         )

# # #                         # ✅ Clean up Google redirects
# # #                         clean_url = extract_direct_url(direct_url)

# # #                         # ✅ Append cleaned data
# # #                         products.append(ProductResult(
# # #                             product_id=item.get('product_id', str(uuid.uuid4())),
# # #                             product_title=item.get('product_title', 'No title'),
# # #                             product_photo=item.get('product_photos', [None])[0]
# # #                                 if item.get('product_photos')
# # #                                 else item.get('product_photo'),
# # #                             product_price=price,
# # #                             product_rating=item.get('product_rating'),
# # #                             product_url=clean_url
# # #                         ))

# # #             # Save search history
# # #             history = SearchHistory(
# # #                 user_id=current_user["id"],
# # #                 query=search_data.query,
# # #                 query_type=search_data.query_type
# # #             )

# # #             history_dict = history.model_dump()
# # #             history_dict['timestamp'] = history_dict['timestamp'].isoformat()
# # #             await db.search_history.insert_one(history_dict)

# # #             response = SearchResponse(
# # #                 category=category,
# # #                 confidence=confidence,
# # #                 products=products
# # #             )
            
# # #             # Debug: Print response to terminal
# # #             logging.info("Search Response:")
# # #             logging.info(f"Query: {search_data.query}")
# # #             logging.info(f"Category: {category} (confidence: {confidence:.3f})")
# # #             logging.info(f"Products found: {len(products)}")
# # #             if products:
# # #                 logging.info("First product sample:")
# # #                 logging.info(f"- Title: {products[0].product_title}")
# # #                 logging.info(f"- Price: {products[0].product_price}")
# # #                 logging.info(f"- URL: {products[0].product_url}")
            
# # #             return response

# # #     except Exception as e:
# # #         logging.error(f"Search error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# # @api_router.post("/search", response_model=SearchResponse)
# # async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
# #     try:
# #         # 1️⃣ Classify user query
# #         category, confidence = classify_query(search_data.query)

# #         # 2️⃣ Helper to clean up URLs
# #         from urllib.parse import urlparse, parse_qs, unquote

# #         def extract_direct_url(google_url: str):
# #             """Extract actual product link if wrapped in Google redirect"""
# #             if not google_url:
# #                 return "#"
# #             if "google.com" in google_url and "url=" in google_url:
# #                 parsed = urlparse(google_url)
# #                 query = parse_qs(parsed.query)
# #                 if "url" in query:
# #                     return unquote(query["url"][0])
# #             return google_url

# #         # 3️⃣ Call RapidAPI
# #         async with httpx.AsyncClient() as client:
# #             headers = {
# #                 "X-RapidAPI-Key": RAPID_API_KEY,
# #                 "X-RapidAPI-Host": RAPID_API_HOST
# #             }
# #             params = {"q": search_data.query, "country": "in", "language": "en"}

# #             logging.info(f"Fetching from RapidAPI: {params['q']}")

# #             response = await client.get(
# #                 f"https://{RAPID_API_HOST}/search-v2",
# #                 headers=headers,
# #                 params=params,
# #                 timeout=15.0
# #             )

# #             products = []

# #             if response.status_code == 200:
# #                 data = response.json()
# #                 logging.info(f"API Status: {data.get('status')}")

# #                 if data.get('status') == 'OK':
# #                     for item in data.get('data', {}).get('products', [])[:12]:
# #                         # ✅ Extract price
# #                         price = None
# #                         if isinstance(item.get('offer'), dict):
# #                             price = item['offer'].get('price')
# #                         elif item.get('typical_price_range'):
# #                             price = item['typical_price_range']

# #                         # ✅ Try to get the cleanest, most direct product URL
# #                         offer = item.get('offer', {})
# #                         direct_url = (
# #                             offer.get('offer_page_url') or
# #                             offer.get('link') or
# #                             item.get('product_page_url') or
# #                             item.get('product_link') or
# #                             item.get('product_url') or
# #                             "#"
# #                         )

# #                         clean_url = extract_direct_url(direct_url)

# #                         # ✅ Add cleaned and structured data
# #                         products.append(ProductResult(
# #                             product_id=item.get('product_id', str(uuid.uuid4())),
# #                             product_title=item.get('product_title', 'No title'),
# #                             product_photo=item.get('product_photos', [None])[0]
# #                                 if item.get('product_photos')
# #                                 else item.get('product_photo'),
# #                             product_price=price,
# #                             product_rating=item.get('product_rating'),
# #                             product_url=clean_url
# #                         ))

# #             # 4️⃣ Save search history
# #             history = SearchHistory(
# #                 user_id=current_user["id"],
# #                 query=search_data.query,
# #                 query_type=search_data.query_type
# #             )
# #             history_dict = history.model_dump()
# #             history_dict['timestamp'] = history_dict['timestamp'].isoformat()
# #             await db.search_history.insert_one(history_dict)

# #             # 5️⃣ Create structured response
# #             response = SearchResponse(
# #                 category=category,
# #                 confidence=confidence,
# #                 products=products
# #             )

# #             # Debug logs
# #             logging.info(f"✅ Query: {search_data.query}")
# #             logging.info(f"Category: {category} ({confidence:.3f})")
# #             logging.info(f"Products returned: {len(products)}")
# #             if products:
# #                 sample = products[0]
# #                 logging.info(f"Sample Product → {sample.product_title} | {sample.product_url}")

# #             return response

# #     except Exception as e:
# #         logging.error(f"❌ Search error: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    

# # @api_router.post("/search", response_model=SearchResponse)
# # async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
# #     try:
# #         # Classify query
# #         category, confidence = classify_query(search_data.query)

# #         # Fetch products from RapidAPI (existing logic)
# #         products = await fetch_products_from_rapidapi(search_data.query)  # abstract your existing logic here

# #         # Apply filters if provided
# #         filtered_products = apply_filters(products, search_data.filters)

# #         # Save search history (existing logic)
# #         await save_search_history(current_user["id"], search_data)

# #         return SearchResponse(
# #             category=category,
# #             confidence=confidence,
# #             products=filtered_products
# #         )

# #     except Exception as e:
# #         logging.error(f"❌ Search error: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# # @api_router.get("/history")
# # async def get_search_history(current_user: dict = Depends(get_current_user)):
# #     history = await db.search_history.find(
# #         {"user_id": current_user["id"]},
# #         {"_id": 0}
# #     ).sort("timestamp", -1).limit(50).to_list(50)
    
# #     # Convert ISO strings back to datetime for response
# #     for item in history:
# #         if isinstance(item['timestamp'], str):
# #             item['timestamp'] = datetime.fromisoformat(item['timestamp'])
    
# #     return history

# # # Include the router in the main app
# # app.include_router(api_router)

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_credentials=True,
# #     allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Configure logging with more detailed format
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s - %(levelname)s - %(message)s',
# #     datefmt='%Y-%m-%d %H:%M:%S'
# # )
# # logger = logging.getLogger(__name__)

# # @app.on_event("shutdown")
# # async def shutdown_db_client():
# #     client.close()
# from fastapi import FastAPI, APIRouter, HTTPException, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from starlette.middleware.cors import CORSMiddleware
# from motor.motor_asyncio import AsyncIOMotorClient
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field, ConfigDict, EmailStr
# from typing import List, Optional
# from passlib.context import CryptContext
# from datetime import datetime, timezone, timedelta
# from pathlib import Path
# import os
# import uuid
# import jwt
# import httpx
# import re
# import logging
# from urllib.parse import urlparse, parse_qs, unquote
# import openai
# from fastapi import Query
# import groq
# import os
# import json as json_lib

# import re
# from openai import OpenAI
# # ----------------------------
# # Load environment variables
# # ----------------------------
# ROOT_DIR = Path(__file__).parent
# load_dotenv(ROOT_DIR / '.env')

# # ----------------------------
# # MongoDB setup
# # ----------------------------
# mongo_url = os.environ['MONGO_URL']
# client = AsyncIOMotorClient(mongo_url)
# db = client[os.environ['DB_NAME']]

# # ----------------------------
# # Security
# # ----------------------------
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# security = HTTPBearer()
# JWT_SECRET = os.environ.get('JWT_SECRET_KEY', 'expense-tracker-secret-key-2024')
# JWT_ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# # ----------------------------
# # Logging
# # ----------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)

# # ----------------------------
# # Fast keyword-based classification
# # ----------------------------
# CATEGORY_KEYWORDS = {
#     "electronics": ["electronics", "electronic", "gadget", "device", "tech", "technology"],
#     "computers": ["computer", "laptop", "pc", "desktop", "monitor", "keyboard", "mouse", "gaming"],
#     "phones": ["phone", "smartphone", "mobile", "iphone", "android", "samsung", "tablet"],
#     "clothing": ["clothing", "clothes", "shirt", "pants", "dress", "jacket", "wear", "fashion"],
#     "shoes": ["shoe", "shoes", "sneaker", "boot", "sandal", "footwear"],
#     "books": ["book", "books", "novel", "reading", "literature", "textbook", "ebook"],
#     "home": ["home", "house", "decor", "decoration", "living", "bedroom", "interior"],
#     "kitchen": ["kitchen", "cooking", "cookware", "utensil", "appliance", "pot", "pan"],
#     "furniture": ["furniture", "chair", "table", "desk", "sofa", "couch", "bed", "cabinet"],
#     "toys": ["toy", "toys", "game", "play", "kids", "children", "puzzle"],
#     "sports": ["sport", "sports", "fitness", "exercise", "gym", "athletic", "workout"],
#     "health": ["health", "medical", "wellness", "care", "medicine", "pain", "therapy"],
#     "beauty": ["beauty", "cosmetic", "makeup", "skincare", "hair", "perfume"],
#     "jewelry": ["jewelry", "jewellery", "ring", "necklace", "bracelet", "watch"],
#     "automotive": ["car", "automotive", "vehicle", "auto", "motorcycle", "bike"]
# }

# def classify_query(query: str):
#     query_lower = query.lower()
#     scores = {}
#     for category, keywords in CATEGORY_KEYWORDS.items():
#         score = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', query_lower))
#         scores[category] = score
#     if max(scores.values()) > 0:
#         best_category = max(scores.items(), key=lambda x: x[1])[0]
#         confidence = min(scores[best_category] * 0.3, 0.95)
#     else:
#         best_category = "electronics"
#         confidence = 0.5
#     return best_category, confidence

# # --- open api key
# OPENAI_API_KEY = "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"
# openai.api_key = OPENAI_API_KEY

# # ----------------------------
# # RapidAPI config
# # ----------------------------
# # RAPID_API_KEY = os.environ.get("RAPID_API_KEY", "YOUR_RAPIDAPI_KEY")
# # RAPID_API_HOST = "real-time-product-search.p.rapidapi.com"
# RAPID_API_KEY = "b46dd21eebmsha1d9dd7586bbe73p167ff5jsnb6e56990e0e7"
# RAPID_API_HOST = "real-time-product-search.p.rapidapi.com"
# # ----------------------------
# # Models
# # ----------------------------
# class User(BaseModel):
#     model_config = ConfigDict(extra="ignore")
#     id: str = Field(default_factory=lambda: str(uuid.uuid4()))
#     email: EmailStr
#     hashed_password: str
#     created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# class UserRegister(BaseModel):
#     email: EmailStr
#     password: str

# class UserLogin(BaseModel):
#     email: EmailStr
#     password: str

# class Token(BaseModel):
#     access_token: str
#     token_type: str
#     user: dict

# class FilterOptions(BaseModel):
#     min_price: Optional[float] = None
#     max_price: Optional[float] = None
#     min_rating: Optional[float] = None
#     category: Optional[str] = None

# class SearchRequest(BaseModel):
#     query: str
#     query_type: str = 'text'
#     filters: Optional[FilterOptions] = None

# class ProductResult(BaseModel):
#     product_id: str
#     product_title: str
#     product_photo: Optional[str]
#     product_price: Optional[str]
#     product_rating: Optional[float]
#     product_url: str

# class BundleResult(BaseModel):
#     title: str
#     description: str
#     products: List[ProductResult]

# # Update SearchResponse to include bundles
# class SearchResponse(BaseModel):
#     category: str
#     confidence: float
#     products: List[ProductResult]
#     bundles: List[BundleResult] = []

# # class SearchResponse(BaseModel):
# #     category: str
# #     confidence: float
# #     products: List[ProductResult]

# class SearchHistory(BaseModel):
#     model_config = ConfigDict(extra="ignore")
#     id: str = Field(default_factory=lambda: str(uuid.uuid4()))
#     user_id: str
#     query: str
#     query_type: str
#     timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# # ----------------------------
# # Helper functions
# # ----------------------------
# def hash_password(password: str) -> str:
#     return pwd_context.hash(password)

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(plain_password, hashed_password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     to_encode = data.copy()
#     expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
#     return encoded_jwt

# async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     try:
#         token = credentials.credentials
#         payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
#         user_id = payload.get("sub")
#         if user_id is None:
#             raise HTTPException(status_code=401, detail="Invalid authentication credentials")
#         user = await db.users.find_one({"id": user_id}, {"_id": 0})
#         if not user:
#             raise HTTPException(status_code=401, detail="User not found")
#         return user
#     except jwt.ExpiredSignatureError:
#         raise HTTPException(status_code=401, detail="Token has expired")
#     except jwt.PyJWTError:
#         raise HTTPException(status_code=401, detail="Could not validate credentials")

# async def fetch_products_from_rapidapi(query: str) -> List[ProductResult]:
#     async with httpx.AsyncClient() as client:
#         headers = {
#             "X-RapidAPI-Key": RAPID_API_KEY,
#             "X-RapidAPI-Host": RAPID_API_HOST
#         }
#         params = {"q": query, "country": "in", "language": "en"}
#         response = await client.get(f"https://{RAPID_API_HOST}/search-v2", headers=headers, params=params, timeout=15.0)
#         products = []

#         if response.status_code == 200:
#             data = response.json()
#             if data.get('status') == 'OK':
#                 def _extract_price(it: dict):
#                     # Try several common places/formats for price
#                     price = None
#                     offer = it.get('offer') or {}
#                     # offer.price may be a dict or string/number
#                     if isinstance(offer, dict):
#                         p = offer.get('price')
#                         if p:
#                             if isinstance(p, dict):
#                                 price = p.get('value') or p.get('amount') or p.get('price')
#                             else:
#                                 price = p

#                     # fallback fields
#                     if not price:
#                         for key in ('typical_price_range', 'price', 'product_price', 'display_price', 'price_range'):
#                             val = it.get(key)
#                             if val:
#                                 price = val
#                                 break

#                     # handle nested numeric values
#                     if isinstance(price, dict):
#                         price = price.get('value') or price.get('amount') or None
#                     if isinstance(price, (int, float)):
#                         price = str(price)
#                     return price

#                 def _extract_image(it: dict):
#                     # Common image fields and nested structures
#                     photos = it.get('product_photos') or it.get('images') or it.get('product_images')
#                     if photos and isinstance(photos, list) and len(photos) > 0:
#                         first = photos[0]
#                         if isinstance(first, dict):
#                             return first.get('url') or first.get('src') or first.get('image')
#                         elif isinstance(first, str):
#                             return first

#                     for key in ('product_photo', 'product_image', 'image', 'thumbnail', 'product_thumb'):
#                         val = it.get(key)
#                         if val:
#                             return val

#                     offer = it.get('offer') or {}
#                     if isinstance(offer, dict):
#                         imgs = offer.get('images') or offer.get('offer_photos') or offer.get('image')
#                         if imgs:
#                             if isinstance(imgs, list) and len(imgs) > 0:
#                                 first = imgs[0]
#                                 if isinstance(first, dict):
#                                     return first.get('url') or first.get('src')
#                                 elif isinstance(first, str):
#                                     return first
#                             elif isinstance(imgs, str):
#                                 return imgs
#                     return None

#                 def _extract_url(it: dict):
#                     offer = it.get('offer', {}) or {}
#                     direct_url = offer.get('offer_page_url') or offer.get('link') or it.get('product_page_url') or it.get('product_link') or it.get('product_url') or "#"
#                     if "google.com" in str(direct_url) and "url=" in str(direct_url):
#                         parsed = urlparse(direct_url)
#                         q = parse_qs(parsed.query)
#                         direct_url = unquote(q.get('url', ['#'])[0])
#                     return direct_url

#                 for item in data.get('data', {}).get('products', [])[:12]:
#                     price = _extract_price(item)
#                     img = _extract_image(item)
#                     direct_url = _extract_url(item)

#                     products.append(ProductResult(
#                         product_id=item.get('product_id', str(uuid.uuid4())),
#                         product_title=item.get('product_title', 'No title'),
#                         product_photo=img,
#                         product_price=price,
#                         product_rating=item.get('product_rating'),
#                         product_url=direct_url
#                     ))
#     return products
# # async def get_complementary_products(base_product: str, budget: Optional[float] = None, max_items: int = 5):
#     """
#     Generate complementary product suggestions for a base product using GPT,
#     then fetch real-time details from RapidAPI and return a list of ProductResult.
#     """
#     prompt = f"""
#     You are an e-commerce assistant. A user is buying '{base_product}'.
#     Suggest up to {max_items} complementary products that fit within a budget of {budget or 'any'} INR.
#     Return only product titles in a JSON array like:
#     ["Gaming Mouse", "Mechanical Keyboard", "Laptop Cooling Pad"]
#     """
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a helpful product recommendation assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7
#         )

#         titles = response['choices'][0]['message']['content'].strip()
#         import json
#         suggested_titles = json.loads(titles)
#         if not isinstance(suggested_titles, list):
#             suggested_titles = [suggested_titles]

#         # Fetch real-time data from RapidAPI for each suggestion
#         complementary_products = []
#         for title in suggested_titles:
#             products = await fetch_products_from_rapidapi(title)
#             # Apply budget filter if given
#             for p in products:
#                 try:
#                     price = float(re.sub(r'[^\d.]', '', str(p.product_price))) if p.product_price else None
#                 except:
#                     price = None
#                 if budget and price and price > budget:
#                     continue
#                 complementary_products.append(p)
#                 # Keep only 1 product per suggestion to reduce clutter
#                 break

#         # Limit total items within budget
#         if budget:
#             selected_products = []
#             total = 0
#             for p in complementary_products:
#                 try:
#                     price = float(re.sub(r'[^\d.]', '', str(p.product_price))) if p.product_price else 0
#                 except:
#                     price = 0
#                 if total + price <= budget:
#                     selected_products.append(p)
#                     total += price
#             return selected_products
#         else:
#             return complementary_products[:max_items]

#     except Exception as e:
#         logger.error(f"Error in AI complementary products: {str(e)}")
#         return []

# # GROQ_API_KEY = "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"
# # client = groq.Client(api_key=GROQ_API_KEY)


# client = OpenAI(
#     api_key="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",
#     base_url="https://api.groq.com/openai/v1",
# )


# async def get_complementary_products(base_product: str, budget: Optional[float] = None, max_items: int = 5):
#     """
#     Generate complementary product suggestions using Groq's OpenAI-compatible API,
#     then fetch real-time product data from RapidAPI.
#     """
#     prompt = f"""
#     You are an e-commerce assistant. A user is buying '{base_product}'.
#     Suggest up to {max_items} complementary products that would go well with it.
#     Keep the total within a budget of {budget or 'any'} INR.
#     Return only a JSON array of product titles, e.g.:
#     ["Gaming Mouse", "Mechanical Keyboard", "Laptop Cooling Pad"]
#     """


#     try:
#         # Groq-compatible call
#         response = client.responses.create(
#             model="openai/gpt-oss-20b",
#             input=[
#                 {"role": "system", "content": "You are a helpful AI assistant that recommends products."},
#                 {"role": "user", "content": prompt},
#             ],
#         )

#         # Debug: print raw Groq response
#         logger.info(f"Groq raw response: {response}")
#         output_text = response.output_text.strip()
#         logger.info(f"Groq output_text: {output_text}")

#         # Parse JSON array safely
#         try:
#             suggested_titles = json_lib.loads(output_text)
#         except Exception as json_err:
#             logger.error(f"JSON parsing error: {json_err}")
#             # fallback if not valid JSON
#             suggested_titles = re.findall(r'"(.*?)"', output_text)

#         if not isinstance(suggested_titles, list):
#             suggested_titles = [suggested_titles]

#         # Fetch each complementary product from RapidAPI
#         complementary_products = []
#         for title in suggested_titles:
#             products = await fetch_products_from_rapidapi(title)
#             for p in products:
#                 try:
#                     price = float(re.sub(r"[^\d.]", "", str(p.product_price))) if p.product_price else None
#                 except Exception as price_err:
#                     logger.error(f"Price parsing error: {price_err}")
#                     price = None
#                 if budget and price and price > budget:
#                     continue
#                 complementary_products.append(p)
#                 break  # one product per title

#         # Optional: fit within total budget
#         if budget:
#             selected = []
#             total = 0
#             for p in complementary_products:
#                 try:
#                     price = float(re.sub(r"[^\d.]", "", str(p.product_price))) if p.product_price else 0
#                 except Exception as price_err:
#                     logger.error(f"Total price parsing error: {price_err}")
#                     price = 0
#                 if total + price <= budget:
#                     selected.append(p)
#                     total += price
#             return selected
#         else:
#             return complementary_products[:max_items]

#     except Exception as e:
#         import traceback
#         logger.error(f"Error in Groq complementary products: {str(e)}\n{traceback.format_exc()}")
#         return []

# # async def get_complementary_products(base_product: str, budget: Optional[float] = None, max_items: int = 5):
#     """
#     Generate complementary product suggestions for a base product using Groq AI,
#     then fetch real-time details from RapidAPI and return a list of ProductResult.
#     """
#     prompt = f"""
#     You are an e-commerce assistant. A user is buying '{base_product}'.
#     Suggest up to {max_items} complementary products that fit within a budget of {budget or 'any'} INR.
#     Return only product titles in a JSON array like:
#     ["Gaming Mouse", "Mechanical Keyboard", "Laptop Cooling Pad"]
#     """

#     try:
#         response = client.completion(
#             model="groq-1", 
#             prompt=prompt,
#             max_tokens=150,
#             temperature=0.7
#         )

#         # Groq returns plain text, parse JSON
#         import json
#         suggested_titles = json.loads(response.choices[0].text.strip())
#         if not isinstance(suggested_titles, list):
#             suggested_titles = [suggested_titles]

#         # Fetch real-time data from RapidAPI for each suggestion
#         complementary_products = []
#         for title in suggested_titles:
#             products = await fetch_products_from_rapidapi(title)
#             for p in products:
#                 try:
#                     price = float(re.sub(r'[^\d.]', '', str(p.product_price))) if p.product_price else None
#                 except:
#                     price = None
#                 if budget and price and price > budget:
#                     continue
#                 complementary_products.append(p)
#                 break  # only one product per suggestion

#         # Limit total items within budget
#         if budget:
#             selected_products = []
#             total = 0
#             for p in complementary_products:
#                 try:
#                     price = float(re.sub(r'[^\d.]', '', str(p.product_price))) if p.product_price else 0
#                 except:
#                     price = 0
#                 if total + price <= budget:
#                     selected_products.append(p)
#                     total += price
#             return selected_products
#         else:
#             return complementary_products[:max_items]

#     except Exception as e:
#         logger.error(f"Error in Groq complementary products: {str(e)}")
#         return []
# def apply_filters(products: List[ProductResult], filters: Optional[FilterOptions]):
#     if not filters:
#         return products
#     filtered = []
#     for p in products:
#         try:
#             price = float(re.sub(r'[^\d.]', '', str(p.product_price))) if p.product_price else None
#         except:
#             price = None
#         if filters.min_price and (price is None or price < filters.min_price):
#             continue
#         if filters.max_price and (price is None or price > filters.max_price):
#             continue
#         if filters.min_rating and (p.product_rating is None or p.product_rating < filters.min_rating):
#             continue
#         if filters.category and filters.category.lower() not in p.product_title.lower():
#             continue
#         filtered.append(p)
#     return filtered

# async def save_search_history(user_id: str, search_data: SearchRequest):
#     history = SearchHistory(user_id=user_id, query=search_data.query, query_type=search_data.query_type)
#     history_dict = history.model_dump()
#     history_dict['timestamp'] = history_dict['timestamp'].isoformat()
#     await db.search_history.insert_one(history_dict)

# # ----------------------------
# # FastAPI setup
# # ----------------------------
# app = FastAPI()
# api_router = APIRouter(prefix="/api")

# # ----------------------------
# # Routes
# # ----------------------------
# @api_router.get("/")
# async def root():
#     return {"message": "NeedMate API is running"}

# @api_router.post("/auth/register", response_model=Token)
# async def register(user_data: UserRegister):
#     existing_user = await db.users.find_one({"email": user_data.email})
#     if existing_user:
#         raise HTTPException(status_code=400, detail="Email already registered")
#     user = User(email=user_data.email, hashed_password=hash_password(user_data.password))
#     await db.users.insert_one(user.model_dump())
#     access_token = create_access_token(data={"sub": user.id})
#     return Token(access_token=access_token, token_type="bearer", user={"id": user.id, "email": user.email})

# @api_router.post("/auth/login", response_model=Token)
# async def login(user_data: UserLogin):
#     user = await db.users.find_one({"email": user_data.email}, {"_id": 0})
#     if not user or not verify_password(user_data.password, user["hashed_password"]):
#         raise HTTPException(status_code=401, detail="Invalid email or password")
#     access_token = create_access_token(data={"sub": user["id"]})
#     return Token(access_token=access_token, token_type="bearer", user={"id": user["id"], "email": user["email"]})

# @api_router.get("/auth/me")
# async def get_me(current_user: dict = Depends(get_current_user)):
#     return {"id": current_user["id"], "email": current_user["email"]}

# # Update the /search endpoint
# @api_router.post("/search", response_model=SearchResponse)
# async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
#     try:
#         # Step 1: Classify the query
#         category, confidence = classify_query(search_data.query)

#         # Step 2: Fetch main products
#         products = await fetch_products_from_rapidapi(search_data.query)
#         filtered_products = apply_filters(products, search_data.filters)

#         # Step 3: Create smart bundles
#         bundles = []
#         if filtered_products:
#             # Create bundles from top products
#             for i, base_product in enumerate(filtered_products[:3]):  # Top 3 products
#                 top_product_title = base_product.product_title
#                 budget = None
#                 if search_data.filters and search_data.filters.max_price:
#                     budget = search_data.filters.max_price
                
#                 # Get complementary products for this base product
#                 complementary = await get_complementary_products(
#                     top_product_title, 
#                     budget, 
#                     max_items=3
#                 )
                
#                 if complementary:
#                     # Create a bundle with the base product + complementary items
#                     bundle = BundleResult(
#                         title=f"Complete {category.title()} Bundle {i+1}",
#                         description=f"Get {base_product.product_title} with perfect complementary items",
#                         products=[base_product] + complementary
#                     )
#                     bundles.append(bundle)

#         # Step 4: Save search history
#         await save_search_history(current_user["id"], search_data)

#         # Step 5: Return results with separate bundles
#         logger.info(f"✅ Returning {len(filtered_products)} products and {len(bundles)} bundles")
        
#         return SearchResponse(
#             category=category, 
#             confidence=confidence, 
#             products=filtered_products,  # ✅ Main products only
#             bundles=bundles  # ✅ Smart bundles separately
#         )

#     except Exception as e:
#         logger.error(f"Search error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# # @api_router.post("/search", response_model=SearchResponse)
# # async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
# #     try:
# #         # Step 1: Classify the query
# #         category, confidence = classify_query(search_data.query)

# #         # Step 2: Fetch main products
# #         products = await fetch_products_from_rapidapi(search_data.query)
# #         filtered_products = apply_filters(products, search_data.filters)

# #         # Step 3: Fetch complementary products (smart bundling)
# #         complementary_products = []
# #         if filtered_products:
# #             # Take top 1 or 2 main products as base for bundling
# #             top_product_title = filtered_products[0].product_title
# #             budget = None
# #             if search_data.filters and search_data.filters.max_price:
# #                 budget = search_data.filters.max_price
# #             complementary_products = await get_complementary_products(top_product_title, budget, max_items=3)

# #         # Combine main + complementary products (optional: tag them differently if needed)
# #         final_products = filtered_products + complementary_products

# #         # Step 4: Save search history
# #         await save_search_history(current_user["id"], search_data)

# #         # Step 5: Return results
# #         return SearchResponse(category=category, confidence=confidence, products=final_products)

# #     except Exception as e:
# #         logger.error(f"Search error: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# # @api_router.post("/search", response_model=SearchResponse)
# # async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
# #     try:
# #         category, confidence = classify_query(search_data.query)
# #         products = await fetch_products_from_rapidapi(search_data.query)
# #         filtered_products = apply_filters(products, search_data.filters)
# #         await save_search_history(current_user["id"], search_data)
# #         return SearchResponse(category=category, confidence=confidence, products=filtered_products)
# #     except Exception as e:
# #         logger.error(f"Search error: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# @api_router.get("/history")
# async def get_search_history(current_user: dict = Depends(get_current_user)):
#     history = await db.search_history.find({"user_id": current_user["id"]}, {"_id": 0}).sort("timestamp", -1).limit(50).to_list(50)
#     for item in history:
#         if isinstance(item['timestamp'], str):
#             item['timestamp'] = datetime.fromisoformat(item['timestamp'])
#     return history

# @api_router.post("/complementary", response_model=List[ProductResult])
# async def complementary_products(
#     product_name: str = Query(..., description="Base product to get complementary items for"),
#     budget: Optional[float] = Query(None, description="Max total budget for complementary products"),
#     current_user: dict = Depends(get_current_user)
# ):
#     """
#     Returns complementary products for a given product within a budget limit.
#     Fetches real-time product data from RapidAPI.
#     """
#     products = await get_complementary_products(product_name, budget)
#     return products

# # ----------------------------
# # Middleware and router
# # ----------------------------
# app.include_router(api_router)
# app.add_middleware(
#     CORSMiddleware,
#     allow_credentials=True,
#     allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ----------------------------
# # Shutdown event
# # ----------------------------
# @app.on_event("shutdown")
# async def shutdown_db_client():
#     client.close()
# from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from dotenv import load_dotenv
# from starlette.middleware.cors import CORSMiddleware
# from motor.motor_asyncio import AsyncIOMotorClient
# import os
# import logging
# from pathlib import Path
# from pydantic import BaseModel, Field, ConfigDict, EmailStr
# from typing import List, Optional
# import uuid
# from datetime import datetime, timezone, timedelta
# from passlib.context import CryptContext
# import jwt
# import httpx
# import re

# ROOT_DIR = Path(__file__).parent
# load_dotenv(ROOT_DIR / '.env')

# # MongoDB connection
# mongo_url = os.environ['MONGO_URL']
# client = AsyncIOMotorClient(mongo_url)
# db = client[os.environ['DB_NAME']]

# # Password hashing
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # JWT settings
# JWT_SECRET = os.environ.get('JWT_SECRET_KEY', 'expense-tracker-secret-key-2024')
# JWT_ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# # Security
# security = HTTPBearer()

# # Fast keyword-based classification (replaces heavy ML model)
# CATEGORY_KEYWORDS = {
#     "electronics": ["electronics", "electronic", "gadget", "device", "tech", "technology"],
#     "computers": ["computer", "laptop", "pc", "desktop", "monitor", "keyboard", "mouse", "gaming"],
#     "phones": ["phone", "smartphone", "mobile", "iphone", "android", "samsung", "tablet"],
#     "clothing": ["clothing", "clothes", "shirt", "pants", "dress", "jacket", "wear", "fashion"],
#     "shoes": ["shoe", "shoes", "sneaker", "boot", "sandal", "footwear"],
#     "books": ["book", "books", "novel", "reading", "literature", "textbook", "ebook"],
#     "home": ["home", "house", "decor", "decoration", "living", "bedroom", "interior"],
#     "kitchen": ["kitchen", "cooking", "cookware", "utensil", "appliance", "pot", "pan"],
#     "furniture": ["furniture", "chair", "table", "desk", "sofa", "couch", "bed", "cabinet"],
#     "toys": ["toy", "toys", "game", "play", "kids", "children", "puzzle"],
#     "sports": ["sport", "sports", "fitness", "exercise", "gym", "athletic", "workout"],
#     "health": ["health", "medical", "wellness", "care", "medicine", "pain", "therapy"],
#     "beauty": ["beauty", "cosmetic", "makeup", "skincare", "hair", "perfume"],
#     "jewelry": ["jewelry", "jewellery", "ring", "necklace", "bracelet", "watch"],
#     "automotive": ["car", "automotive", "vehicle", "auto", "motorcycle", "bike"]
# }

# def classify_query(query: str):
#     """Fast keyword-based classification"""
#     query_lower = query.lower()
#     scores = {}
    
#     for category, keywords in CATEGORY_KEYWORDS.items():
#         score = 0
#         for keyword in keywords:
#             if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
#                 score += 1
#         scores[category] = score
    
#     # Get category with highest score
#     if max(scores.values()) > 0:
#         best_category = max(scores.items(), key=lambda x: x[1])[0]
#         confidence = min(scores[best_category] * 0.3, 0.95)  # Scale to confidence
#     else:
#         # Default to general search
#         best_category = "electronics"
#         confidence = 0.5
    
#     return best_category, confidence

# # RapidAPI settings
# RAPID_API_KEY = "3227ea0400mshbc7f9fb81508577p189076jsnf5b6c369e941"
# RAPID_API_HOST = "real-time-product-search.p.rapidapi.com"

# # Create the main app without a prefix
# app = FastAPI()

# # Create a router with the /api prefix
# api_router = APIRouter(prefix="/api")

# # Models
# class User(BaseModel):
#     model_config = ConfigDict(extra="ignore")
#     id: str = Field(default_factory=lambda: str(uuid.uuid4()))
#     email: EmailStr
#     hashed_password: str
#     created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# class UserRegister(BaseModel):
#     email: EmailStr
#     password: str

# class UserLogin(BaseModel):
#     email: EmailStr
#     password: str

# class Token(BaseModel):
#     access_token: str
#     token_type: str
#     user: dict

# class FilterOptions(BaseModel):
#     min_price: Optional[float] = None
#     max_price: Optional[float] = None
#     min_rating: Optional[float] = None
#     category: Optional[str] = None  # To filter by category


# class SearchHistory(BaseModel):
#     model_config = ConfigDict(extra="ignore")
#     id: str = Field(default_factory=lambda: str(uuid.uuid4()))
#     user_id: str
#     query: str
#     query_type: str  # 'text' or 'voice'
#     timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# # class SearchRequest(BaseModel):
# #     query: str
# #     query_type: str = 'text'
# class SearchRequest(BaseModel):
#     query: str
#     query_type: str = 'text'
#     filters: Optional[FilterOptions] = None

# class ProductResult(BaseModel):
#     product_id: str
#     product_title: str
#     product_photo: Optional[str]
#     product_price: Optional[str]
#     product_rating: Optional[float]
#     product_url: str

# class SearchResponse(BaseModel):
#     category: str
#     confidence: float
#     products: List[ProductResult]

# # appliying filters

# def apply_filters(products: List[ProductResult], filters: Optional[FilterOptions]):
#     if not filters:
#         return products

#     filtered = []

#     for product in products:
#         # Price filtering
#         price = None
#         if product.product_price:
#             try:
#                 price = float(re.sub(r'[^\d.]', '', str(product.product_price)))
#             except:
#                 price = None

#         if filters.min_price and (price is None or price < filters.min_price):
#             continue
#         if filters.max_price and (price is None or price > filters.max_price):
#             continue

#         # Rating filtering
#         if filters.min_rating and (product.product_rating is None or product.product_rating < filters.min_rating):
#             continue

#         # Category filtering
#         if filters.category and filters.category.lower() not in product.product_title.lower():
#             continue

#         filtered.append(product)

#     return filtered

# # Helper functions
# def hash_password(password: str) -> str:
#     return pwd_context.hash(password)

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(plain_password, hashed_password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.now(timezone.utc) + expires_delta
#     else:
#         expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
#     return encoded_jwt

# async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     try:
#         token = credentials.credentials
#         payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
#         user_id: str = payload.get("sub")
#         if user_id is None:
#             raise HTTPException(status_code=401, detail="Invalid authentication credentials")
#         user = await db.users.find_one({"id": user_id}, {"_id": 0})
#         if user is None:
#             raise HTTPException(status_code=401, detail="User not found")
#         return user
#     except jwt.ExpiredSignatureError:
#         raise HTTPException(status_code=401, detail="Token has expired")
#     except jwt.PyJWTError:
#         raise HTTPException(status_code=401, detail="Could not validate credentials")

# # Routes
# @api_router.get("/")
# async def root():
#     return {"message": "NeedMate API is running"}

# @api_router.post("/auth/register", response_model=Token)
# async def register(user_data: UserRegister):
#     # Check if user exists
#     existing_user = await db.users.find_one({"email": user_data.email})
#     if existing_user:
#         raise HTTPException(status_code=400, detail="Email already registered")
    
#     # Create new user
#     user = User(
#         email=user_data.email,
#         hashed_password=hash_password(user_data.password)
#     )
    
#     user_dict = user.model_dump()
#     user_dict['created_at'] = user_dict['created_at'].isoformat()
    
#     await db.users.insert_one(user_dict)
    
#     # Create access token
#     access_token = create_access_token(data={"sub": user.id})
    
#     return Token(
#         access_token=access_token,
#         token_type="bearer",
#         user={"id": user.id, "email": user.email}
#     )

# @api_router.post("/auth/login", response_model=Token)
# async def login(user_data: UserLogin):
#     # Find user
#     user = await db.users.find_one({"email": user_data.email}, {"_id": 0})
#     if not user:
#         raise HTTPException(status_code=401, detail="Invalid email or password")
    
#     # Verify password
#     if not verify_password(user_data.password, user["hashed_password"]):
#         raise HTTPException(status_code=401, detail="Invalid email or password")
    
#     # Create access token
#     access_token = create_access_token(data={"sub": user["id"]})
    
#     return Token(
#         access_token=access_token,
#         token_type="bearer",
#         user={"id": user["id"], "email": user["email"]}
#     )

# @api_router.get("/auth/me")
# async def get_me(current_user: dict = Depends(get_current_user)):
#     return {"id": current_user["id"], "email": current_user["email"]}

# # @api_router.post("/search", response_model=SearchResponse)
# # async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
# #     try:
# #         # Fast keyword-based classification
# #         category, confidence = classify_query(search_data.query)
        
# #         # Search products using RapidAPI
# #         async with httpx.AsyncClient() as client:
# #             headers = {
# #                 "X-RapidAPI-Key": RAPID_API_KEY,
# #                 "X-RapidAPI-Host": RAPID_API_HOST
# #             }
            
# #             params = {
# #                 "q": search_data.query,
# #                 "country": "in",
# #                 "language": "en"
# #             }
            
# #             response = await client.get(
# #                 f"https://{RAPID_API_HOST}/search-v2",
# #                 headers=headers,
# #                 params=params,
# #                 timeout=15.0
# #             )
            
# #             if response.status_code == 200:
# #                 data = response.json()
# #                 products = []
                
# #                 # Parse products from response
# #                 if data.get('status') == 'OK':
# #                     response_data = data.get('data', {})
# #                     # Get products from the nested structure
# #                     product_data = response_data.get('products', []) if isinstance(response_data, dict) else []
                    
# #                     # Limit to 12 products
# #                     for item in (product_data if isinstance(product_data, list) else [])[:12]:
# #                         # Extract price
# #                         price = None
# #                         if isinstance(item.get('offer'), dict):
# #                             price = item['offer'].get('price')
# #                         elif item.get('typical_price_range'):
# #                             price = item.get('typical_price_range')
                        
# #                         products.append(ProductResult(
# #                             product_id=item.get('product_id', str(uuid.uuid4())),
# #                             product_title=item.get('product_title', 'No title'),
# #                             product_photo=item.get('product_photos', [None])[0] if item.get('product_photos') else item.get('product_photo'),
# #                             product_price=price,
# #                             product_rating=item.get('product_rating'),
# #                             product_url=item.get('product_page_url', '#')
# #                         ))
# #                 else:
# #                     products = []
# #             else:
# #                 products = []
        
# #         # Save search history
# #         history = SearchHistory(
# #             user_id=current_user["id"],
# #             query=search_data.query,
# #             query_type=search_data.query_type
# #         )
        
# #         history_dict = history.model_dump()
# #         history_dict['timestamp'] = history_dict['timestamp'].isoformat()
# #         await db.search_history.insert_one(history_dict)
        
# #         return SearchResponse(
# #             category=category,
# #             confidence=confidence,
# #             products=products
# #         )
    
# #     except Exception as e:
# #         logging.error(f"Search error: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# # @api_router.post("/search", response_model=SearchResponse)
# # async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
# #     try:
# #         # Fast keyword-based classification
# #         category, confidence = classify_query(search_data.query)

# #         # Helper function to extract direct URLs
# #         from urllib.parse import urlparse, parse_qs, unquote

# #         def extract_direct_url(google_url: str):
# #             """Extracts the actual product link if it's wrapped in a Google redirect"""
# #             if not google_url:
# #                 return "#"
# #             if "google.com" in google_url and "url=" in google_url:
# #                 parsed = urlparse(google_url)
# #                 query = parse_qs(parsed.query)
# #                 if "url" in query:
# #                     return unquote(query["url"][0])
# #             return google_url

# #         # Search products using RapidAPI
# #         async with httpx.AsyncClient() as client:
# #             headers = {
# #                 "X-RapidAPI-Key": RAPID_API_KEY,
# #                 "X-RapidAPI-Host": RAPID_API_HOST
# #             }
# #             params = {
# #                 "q": search_data.query,
# #                 "country": "in",
# #                 "language": "en"
# #             }

# #             # Log request details
# #             logging.info(f"API Request URL: https://{RAPID_API_HOST}/search-v2")
# #             logging.info(f"Search Query: {params['q']}")
            
# #             response = await client.get(
# #                 f"https://{RAPID_API_HOST}/search-v2",
# #                 headers=headers,
# #                 params=params,
# #                 timeout=15.0
# #             )
            
# #             logging.info(f"API Status Code: {response.status_code}")

# #             products = []

# #             if response.status_code == 200:
# #                 data = response.json()
# #                 # Print raw API response
# #                 logging.info("Raw API Response:")
# #                 logging.info(f"{data}")
                
# #                 if data.get('status') == 'OK':
# #                     response_data = data.get('data', {})
# #                     product_data = response_data.get('products', [])

# #                     for item in (product_data if isinstance(product_data, list) else [])[:12]:
# #                         # ✅ Extract price safely
# #                         price = None
# #                         if isinstance(item.get('offer'), dict):
# #                             price = item['offer'].get('price')
# #                         elif item.get('typical_price_range'):
# #                             price = item.get('typical_price_range')

# #                         # ✅ Find the most direct product link available
# #                         direct_url = (
# #                             item.get('product_link') or
# #                             item.get('offer', {}).get('link') or
# #                             item.get('product_page_url') or
# #                             item.get('product_url') or
# #                             "#"
# #                         )

# #                         # ✅ Clean up Google redirects
# #                         clean_url = extract_direct_url(direct_url)

# #                         # ✅ Append cleaned data
# #                         products.append(ProductResult(
# #                             product_id=item.get('product_id', str(uuid.uuid4())),
# #                             product_title=item.get('product_title', 'No title'),
# #                             product_photo=item.get('product_photos', [None])[0]
# #                                 if item.get('product_photos')
# #                                 else item.get('product_photo'),
# #                             product_price=price,
# #                             product_rating=item.get('product_rating'),
# #                             product_url=clean_url
# #                         ))

# #             # Save search history
# #             history = SearchHistory(
# #                 user_id=current_user["id"],
# #                 query=search_data.query,
# #                 query_type=search_data.query_type
# #             )

# #             history_dict = history.model_dump()
# #             history_dict['timestamp'] = history_dict['timestamp'].isoformat()
# #             await db.search_history.insert_one(history_dict)

# #             response = SearchResponse(
# #                 category=category,
# #                 confidence=confidence,
# #                 products=products
# #             )
            
# #             # Debug: Print response to terminal
# #             logging.info("Search Response:")
# #             logging.info(f"Query: {search_data.query}")
# #             logging.info(f"Category: {category} (confidence: {confidence:.3f})")
# #             logging.info(f"Products found: {len(products)}")
# #             if products:
# #                 logging.info("First product sample:")
# #                 logging.info(f"- Title: {products[0].product_title}")
# #                 logging.info(f"- Price: {products[0].product_price}")
# #                 logging.info(f"- URL: {products[0].product_url}")
            
# #             return response

# #     except Exception as e:
# #         logging.error(f"Search error: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# @api_router.post("/search", response_model=SearchResponse)
# async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
#     try:
#         # 1️⃣ Classify user query
#         category, confidence = classify_query(search_data.query)

#         # 2️⃣ Helper to clean up URLs
#         from urllib.parse import urlparse, parse_qs, unquote

#         def extract_direct_url(google_url: str):
#             """Extract actual product link if wrapped in Google redirect"""
#             if not google_url:
#                 return "#"
#             if "google.com" in google_url and "url=" in google_url:
#                 parsed = urlparse(google_url)
#                 query = parse_qs(parsed.query)
#                 if "url" in query:
#                     return unquote(query["url"][0])
#             return google_url

#         # 3️⃣ Call RapidAPI
#         async with httpx.AsyncClient() as client:
#             headers = {
#                 "X-RapidAPI-Key": RAPID_API_KEY,
#                 "X-RapidAPI-Host": RAPID_API_HOST
#             }
#             params = {"q": search_data.query, "country": "in", "language": "en"}

#             logging.info(f"Fetching from RapidAPI: {params['q']}")

#             response = await client.get(
#                 f"https://{RAPID_API_HOST}/search-v2",
#                 headers=headers,
#                 params=params,
#                 timeout=15.0
#             )

#             products = []

#             if response.status_code == 200:
#                 data = response.json()
#                 logging.info(f"API Status: {data.get('status')}")

#                 if data.get('status') == 'OK':
#                     for item in data.get('data', {}).get('products', [])[:12]:
#                         # ✅ Extract price
#                         price = None
#                         if isinstance(item.get('offer'), dict):
#                             price = item['offer'].get('price')
#                         elif item.get('typical_price_range'):
#                             price = item['typical_price_range']

#                         # ✅ Try to get the cleanest, most direct product URL
#                         offer = item.get('offer', {})
#                         direct_url = (
#                             offer.get('offer_page_url') or
#                             offer.get('link') or
#                             item.get('product_page_url') or
#                             item.get('product_link') or
#                             item.get('product_url') or
#                             "#"
#                         )

#                         clean_url = extract_direct_url(direct_url)

#                         # ✅ Add cleaned and structured data
#                         products.append(ProductResult(
#                             product_id=item.get('product_id', str(uuid.uuid4())),
#                             product_title=item.get('product_title', 'No title'),
#                             product_photo=item.get('product_photos', [None])[0]
#                                 if item.get('product_photos')
#                                 else item.get('product_photo'),
#                             product_price=price,
#                             product_rating=item.get('product_rating'),
#                             product_url=clean_url
#                         ))

#             # 4️⃣ Save search history
#             history = SearchHistory(
#                 user_id=current_user["id"],
#                 query=search_data.query,
#                 query_type=search_data.query_type
#             )
#             history_dict = history.model_dump()
#             history_dict['timestamp'] = history_dict['timestamp'].isoformat()
#             await db.search_history.insert_one(history_dict)

#             # 5️⃣ Create structured response
#             response = SearchResponse(
#                 category=category,
#                 confidence=confidence,
#                 products=products
#             )

#             # Debug logs
#             logging.info(f"✅ Query: {search_data.query}")
#             logging.info(f"Category: {category} ({confidence:.3f})")
#             logging.info(f"Products returned: {len(products)}")
#             if products:
#                 sample = products[0]
#                 logging.info(f"Sample Product → {sample.product_title} | {sample.product_url}")

#             return response

#     except Exception as e:
#         logging.error(f"❌ Search error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    

# @api_router.post("/search", response_model=SearchResponse)
# async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
#     try:
#         # Classify query
#         category, confidence = classify_query(search_data.query)

#         # Fetch products from RapidAPI (existing logic)
#         products = await fetch_products_from_rapidapi(search_data.query)  # abstract your existing logic here

#         # Apply filters if provided
#         filtered_products = apply_filters(products, search_data.filters)

#         # Save search history (existing logic)
#         await save_search_history(current_user["id"], search_data)

#         return SearchResponse(
#             category=category,
#             confidence=confidence,
#             products=filtered_products
#         )

#     except Exception as e:
#         logging.error(f"❌ Search error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# @api_router.get("/history")
# async def get_search_history(current_user: dict = Depends(get_current_user)):
#     history = await db.search_history.find(
#         {"user_id": current_user["id"]},
#         {"_id": 0}
#     ).sort("timestamp", -1).limit(50).to_list(50)
    
#     # Convert ISO strings back to datetime for response
#     for item in history:
#         if isinstance(item['timestamp'], str):
#             item['timestamp'] = datetime.fromisoformat(item['timestamp'])
    
#     return history

# # Include the router in the main app
# app.include_router(api_router)

# app.add_middleware(
#     CORSMiddleware,
#     allow_credentials=True,
#     allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configure logging with more detailed format
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)

# @app.on_event("shutdown")
# async def shutdown_db_client():
#     client.close()
from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
from passlib.context import CryptContext
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import uuid
import jwt
import httpx
import re
import logging
from urllib.parse import urlparse, parse_qs, unquote
import openai
from fastapi import Query
import groq
import os
import json as json_lib

import re
from openai import OpenAI
# ----------------------------
# Load environment variables
# ----------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# ----------------------------
# MongoDB setup
# ----------------------------
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# ----------------------------
# Security
# ----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
JWT_SECRET = os.environ.get('JWT_SECRET_KEY', 'expense-tracker-secret-key-2024')
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ----------------------------
# Fast keyword-based classification
# ----------------------------
CATEGORY_KEYWORDS = {
    "electronics": ["electronics", "electronic", "gadget", "device", "tech", "technology"],
    "computers": ["computer", "laptop", "pc", "desktop", "monitor", "keyboard", "mouse", "gaming"],
    "phones": ["phone", "smartphone", "mobile", "iphone", "android", "samsung", "tablet"],
    "clothing": ["clothing", "clothes", "shirt", "pants", "dress", "jacket", "wear", "fashion"],
    "shoes": ["shoe", "shoes", "sneaker", "boot", "sandal", "footwear"],
    "books": ["book", "books", "novel", "reading", "literature", "textbook", "ebook"],
    "home": ["home", "house", "decor", "decoration", "living", "bedroom", "interior"],
    "kitchen": ["kitchen", "cooking", "cookware", "utensil", "appliance", "pot", "pan"],
    "furniture": ["furniture", "chair", "table", "desk", "sofa", "couch", "bed", "cabinet"],
    "toys": ["toy", "toys", "game", "play", "kids", "children", "puzzle"],
    "sports": ["sport", "sports", "fitness", "exercise", "gym", "athletic", "workout"],
    "health": ["health", "medical", "wellness", "care", "medicine", "pain", "therapy"],
    "beauty": ["beauty", "cosmetic", "makeup", "skincare", "hair", "perfume"],
    "jewelry": ["jewelry", "jewellery", "ring", "necklace", "bracelet", "watch"],
    "automotive": ["car", "automotive", "vehicle", "auto", "motorcycle", "bike"]
}

def classify_query(query: str):
    query_lower = query.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', query_lower))
        scores[category] = score
    if max(scores.values()) > 0:
        best_category = max(scores.items(), key=lambda x: x[1])[0]
        confidence = min(scores[best_category] * 0.3, 0.95)
    else:
        best_category = "electronics"
        confidence = 0.5
    return best_category, confidence

# --- open api key
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# ----------------------------
# RapidAPI config
# ----------------------------
# RAPID_API_KEY = os.environ.get("RAPID_API_KEY", "YOUR_RAPIDAPI_KEY")
# RAPID_API_HOST = "real-time-product-search.p.rapidapi.com"
RAPID_API_KEY = "b46dd21eebmsha1d9dd7586bbe73p167ff5jsnb6e56990e0e7"
RAPID_API_HOST = "real-time-product-search.p.rapidapi.com"
# ----------------------------
# Models
# ----------------------------
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    hashed_password: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserRegister(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class FilterOptions(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    category: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    query_type: str = 'text'
    filters: Optional[FilterOptions] = None

class ProductResult(BaseModel):
    product_id: str
    product_title: str
    product_photo: Optional[str]
    product_price: Optional[str]
    product_rating: Optional[float]
    product_url: str

class BundleResult(BaseModel):
    title: str
    description: str
    products: List[ProductResult]

# Update SearchResponse to include bundles
class SearchResponse(BaseModel):
    category: str
    confidence: float
    products: List[ProductResult]
    bundles: List[BundleResult] = []

# class SearchResponse(BaseModel):
#     category: str
#     confidence: float
#     products: List[ProductResult]

class SearchHistory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    query: str
    query_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ----------------------------
# Helper functions
# ----------------------------
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        user = await db.users.find_one({"id": user_id}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

async def fetch_products_from_rapidapi(query: str) -> List[ProductResult]:
    async with httpx.AsyncClient() as client:
        headers = {
            "X-RapidAPI-Key": RAPID_API_KEY,
            "X-RapidAPI-Host": RAPID_API_HOST
        }
        params = {"q": query, "country": "in", "language": "en"}
        response = await client.get(f"https://{RAPID_API_HOST}/search-v2", headers=headers, params=params, timeout=15.0)
        products = []

        if response.status_code == 200:
            data = response.json()
            # Defensive checks in case RapidAPI returns unexpected payloads
            if not data or not isinstance(data, dict):
                logger.error(f"RapidAPI returned unexpected response for query '{query}': {data}")
                return products

            products_data = data.get('data') or {}
            product_list = products_data.get('products') if isinstance(products_data, dict) else []

            def _extract_price(it: dict):
                if not isinstance(it, dict):
                    return None
                price = None
                offer = it.get('offer') or {}
                if isinstance(offer, dict):
                    p = offer.get('price')
                    if p:
                        if isinstance(p, dict):
                            price = p.get('value') or p.get('amount') or p.get('price')
                        else:
                            price = p
                if not price:
                    for key in ('typical_price_range', 'price', 'product_price', 'display_price', 'price_range'):
                        val = it.get(key)
                        if val:
                            price = val
                            break
                if isinstance(price, dict):
                    price = price.get('value') or price.get('amount') or None
                if isinstance(price, (int, float)):
                    price = str(price)
                return price

            def _extract_image(it: dict):
                if not isinstance(it, dict):
                    return None
                photos = it.get('product_photos') or it.get('images') or it.get('product_images')
                if photos and isinstance(photos, list) and len(photos) > 0:
                    first = photos[0]
                    if isinstance(first, dict):
                        return first.get('url') or first.get('src') or first.get('image')
                    elif isinstance(first, str):
                        return first
                for key in ('product_photo', 'product_image', 'image', 'thumbnail', 'product_thumb'):
                    val = it.get(key)
                    if val:
                        return val
                offer = it.get('offer') or {}
                if isinstance(offer, dict):
                    imgs = offer.get('images') or offer.get('offer_photos') or offer.get('image')
                    if imgs:
                        if isinstance(imgs, list) and len(imgs) > 0:
                            first = imgs[0]
                            if isinstance(first, dict):
                                return first.get('url') or first.get('src')
                            elif isinstance(first, str):
                                return first
                        elif isinstance(imgs, str):
                            return imgs
                return None

            def _extract_url(it: dict):
                if not isinstance(it, dict):
                    return "#"
                offer = it.get('offer', {}) or {}
                direct_url = offer.get('offer_page_url') or offer.get('link') or it.get('product_page_url') or it.get('product_link') or it.get('product_url') or "#"
                if isinstance(direct_url, str) and "google.com" in direct_url and "url=" in direct_url:
                    parsed = urlparse(direct_url)
                    q = parse_qs(parsed.query)
                    direct_url = unquote(q.get('url', ['#'])[0])
                return direct_url

            for item in (product_list or [])[:12]:
                if not item or not isinstance(item, dict):
                    continue
                
                # Skip items with missing title or price
                title = item.get('product_title')
                if not title or title.strip() == '' or title == 'No title':
                    logger.debug(f"Skipping product with missing/empty title")
                    continue
                
                price = _extract_price(item)
                if not price:
                    logger.debug(f"Skipping product '{title}' with missing price")
                    continue
                
                img = _extract_image(item)
                direct_url = _extract_url(item)

                products.append(ProductResult(
                    product_id=item.get('product_id', str(uuid.uuid4())),
                    product_title=title,
                    product_photo=img,
                    product_price=price,
                    product_rating=item.get('product_rating'),
                    product_url=direct_url
                ))
    return products
# async def get_complementary_products(base_product: str, budget: Optional[float] = None, max_items: int = 5):
    """
    Generate complementary product suggestions for a base product using GPT,
    then fetch real-time details from RapidAPI and return a list of ProductResult.
    """
    prompt = f"""
    You are an e-commerce assistant. A user is buying '{base_product}'.
    Suggest up to {max_items} complementary products that fit within a budget of {budget or 'any'} INR.
    Return only product titles in a JSON array like:
    ["Gaming Mouse", "Mechanical Keyboard", "Laptop Cooling Pad"]
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful product recommendation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        titles = response['choices'][0]['message']['content'].strip()
        import json
        suggested_titles = json.loads(titles)
        if not isinstance(suggested_titles, list):
            suggested_titles = [suggested_titles]

        # Fetch real-time data from RapidAPI for each suggestion
        complementary_products = []
        for title in suggested_titles:
            products = await fetch_products_from_rapidapi(title)
            # Apply budget filter if given
            for p in products:
                try:
                    price = float(re.sub(r'[^\d.]', '', str(p.product_price))) if p.product_price else None
                except:
                    price = None
                if budget and price and price > budget:
                    continue
                complementary_products.append(p)
                # Keep only 1 product per suggestion to reduce clutter
                break

        # Limit total items within budget
        if budget:
            selected_products = []
            total = 0
            for p in complementary_products:
                try:
                    price = float(re.sub(r'[^\d.]', '', str(p.product_price))) if p.product_price else 0
                except:
                    price = 0
                if total + price <= budget:
                    selected_products.append(p)
                    total += price
            return selected_products
        else:
            return complementary_products[:max_items]

    except Exception as e:
        logger.error(f"Error in AI complementary products: {str(e)}")
        return []

# GROQ_API_KEY = "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"
# client = groq.Client(api_key=GROQ_API_KEY)





async def get_complementary_products(base_product: str, budget: Optional[float] = None, max_items: int = 5):
    """
    Generate complementary product suggestions using Groq's OpenAI-compatible API,
    then fetch real-time product data from RapidAPI.
    """
    prompt = f"""
    You are an e-commerce assistant. A user is buying '{base_product}'.
    Suggest up to {max_items} complementary products that would go well with it.
    Keep the total within a budget of {budget or 'any'} INR.
    Return only a JSON array of product titles, e.g.:
    ["Gaming Mouse", "Mechanical Keyboard", "Laptop Cooling Pad"]
    """


    try:
        # Groq-compatible call
        response = client.responses.create(
            model="openai/gpt-oss-20b",
            input=[
                {"role": "system", "content": "You are a helpful AI assistant that recommends products."},
                {"role": "user", "content": prompt},
            ],
        )

        # Debug: print raw Groq response
        logger.info(f"Groq raw response: {response}")
        output_text = response.output_text.strip()
        logger.info(f"Groq output_text: {output_text}")

        # Parse JSON array safely
        try:
            suggested_titles = json_lib.loads(output_text)
        except Exception as json_err:
            logger.error(f"JSON parsing error: {json_err}")
            # fallback if not valid JSON
            suggested_titles = re.findall(r'"(.*?)"', output_text)

        if not isinstance(suggested_titles, list):
            suggested_titles = [suggested_titles]

        # Fetch each complementary product from RapidAPI
        complementary_products = []
        for title in suggested_titles:
            products = await fetch_products_from_rapidapi(title)
            for p in products:
                try:
                    price = float(re.sub(r"[^\d.]", "", str(p.product_price))) if p.product_price else None
                except Exception as price_err:
                    logger.error(f"Price parsing error: {price_err}")
                    price = None
                if budget and price and price > budget:
                    continue
                complementary_products.append(p)
                break  # one product per title

        # Optional: fit within total budget
        if budget:
            selected = []
            total = 0
            for p in complementary_products:
                try:
                    price = float(re.sub(r"[^\d.]", "", str(p.product_price))) if p.product_price else 0
                except Exception as price_err:
                    logger.error(f"Total price parsing error: {price_err}")
                    price = 0
                if total + price <= budget:
                    selected.append(p)
                    total += price
            return selected
        else:
            return complementary_products[:max_items]

    except Exception as e:
        import traceback
        logger.error(f"Error in Groq complementary products: {str(e)}\n{traceback.format_exc()}")
        return []

# async def get_complementary_products(base_product: str, budget: Optional[float] = None, max_items: int = 5):
    """
    Generate complementary product suggestions for a base product using Groq AI,
    then fetch real-time details from RapidAPI and return a list of ProductResult.
    """
    prompt = f"""
    You are an e-commerce assistant. A user is buying '{base_product}'.
    Suggest up to {max_items} complementary products that fit within a budget of {budget or 'any'} INR.
    Return only product titles in a JSON array like:
    ["Gaming Mouse", "Mechanical Keyboard", "Laptop Cooling Pad"]
    """

    try:
        response = client.completion(
            model="groq-1", 
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )

        # Groq returns plain text, parse JSON
        import json
        suggested_titles = json.loads(response.choices[0].text.strip())
        if not isinstance(suggested_titles, list):
            suggested_titles = [suggested_titles]

        # Fetch real-time data from RapidAPI for each suggestion
        complementary_products = []
        for title in suggested_titles:
            products = await fetch_products_from_rapidapi(title)
            for p in products:
                try:
                    price = float(re.sub(r'[^\d.]', '', str(p.product_price))) if p.product_price else None
                except:
                    price = None
                if budget and price and price > budget:
                    continue
                complementary_products.append(p)
                break  # only one product per suggestion

        # Limit total items within budget
        if budget:
            selected_products = []
            total = 0
            for p in complementary_products:
                try:
                    price = float(re.sub(r'[^\d.]', '', str(p.product_price))) if p.product_price else 0
                except:
                    price = 0
                if total + price <= budget:
                    selected_products.append(p)
                    total += price
            return selected_products
        else:
            return complementary_products[:max_items]

    except Exception as e:
        logger.error(f"Error in Groq complementary products: {str(e)}")
        return []
def apply_filters(products: List[ProductResult], filters: Optional[FilterOptions]):
    if not filters:
        return products
    filtered = []
    for p in products:
        try:
            price = float(re.sub(r'[^\d.]', '', str(p.product_price))) if p.product_price else None
        except:
            price = None
        if filters.min_price and (price is None or price < filters.min_price):
            continue
        if filters.max_price and (price is None or price > filters.max_price):
            continue
        if filters.min_rating and (p.product_rating is None or p.product_rating < filters.min_rating):
            continue
        if filters.category and filters.category.lower() not in p.product_title.lower():
            continue
        filtered.append(p)
    return filtered

async def save_search_history(user_id: str, search_data: SearchRequest):
    history = SearchHistory(user_id=user_id, query=search_data.query, query_type=search_data.query_type)
    history_dict = history.model_dump()
    history_dict['timestamp'] = history_dict['timestamp'].isoformat()
    await db.search_history.insert_one(history_dict)

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI()
api_router = APIRouter(prefix="/api")

# ----------------------------
# Routes
# ----------------------------
@api_router.get("/")
async def root():
    return {"message": "NeedMate API is running"}

@api_router.post("/auth/register", response_model=Token)
async def register(user_data: UserRegister):
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=user_data.email, hashed_password=hash_password(user_data.password))
    await db.users.insert_one(user.model_dump())
    access_token = create_access_token(data={"sub": user.id})
    return Token(access_token=access_token, token_type="bearer", user={"id": user.id, "email": user.email})

@api_router.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    user = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    access_token = create_access_token(data={"sub": user["id"]})
    return Token(access_token=access_token, token_type="bearer", user={"id": user["id"], "email": user["email"]})

@api_router.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {"id": current_user["id"], "email": current_user["email"]}

# Update the /search endpoint
@api_router.post("/search", response_model=SearchResponse)
async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
    try:
        # Step 1: Classify the query
        category, confidence = classify_query(search_data.query)

        # Step 2: Fetch main products
        products = await fetch_products_from_rapidapi(search_data.query)
        filtered_products = apply_filters(products, search_data.filters)

        # Step 3: Create smart bundles
        bundles = []
        if filtered_products:
            # Create bundles from top products
            for i, base_product in enumerate(filtered_products[:3]):  # Top 3 products
                top_product_title = base_product.product_title
                budget = None
                if search_data.filters and search_data.filters.max_price:
                    budget = search_data.filters.max_price
                
                # Get complementary products for this base product
                complementary = await get_complementary_products(
                    top_product_title, 
                    budget, 
                    max_items=3
                )
                
                if complementary:
                    # Create a bundle with the base product + complementary items
                    bundle = BundleResult(
                        title=f"Complete {category.title()} Bundle {i+1}",
                        description=f"Get {base_product.product_title} with perfect complementary items",
                        products=[base_product] + complementary
                    )
                    bundles.append(bundle)

        # Step 4: Save search history
        await save_search_history(current_user["id"], search_data)

        # Step 5: Return results with separate bundles
        logger.info(f"✅ Returning {len(filtered_products)} products and {len(bundles)} bundles")
        
        return SearchResponse(
            category=category, 
            confidence=confidence, 
            products=filtered_products,  # ✅ Main products only
            bundles=bundles  # ✅ Smart bundles separately
        )

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# @api_router.post("/search", response_model=SearchResponse)
# async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
#     try:
#         # Step 1: Classify the query
#         category, confidence = classify_query(search_data.query)

#         # Step 2: Fetch main products
#         products = await fetch_products_from_rapidapi(search_data.query)
#         filtered_products = apply_filters(products, search_data.filters)

#         # Step 3: Fetch complementary products (smart bundling)
#         complementary_products = []
#         if filtered_products:
#             # Take top 1 or 2 main products as base for bundling
#             top_product_title = filtered_products[0].product_title
#             budget = None
#             if search_data.filters and search_data.filters.max_price:
#                 budget = search_data.filters.max_price
#             complementary_products = await get_complementary_products(top_product_title, budget, max_items=3)

#         # Combine main + complementary products (optional: tag them differently if needed)
#         final_products = filtered_products + complementary_products

#         # Step 4: Save search history
#         await save_search_history(current_user["id"], search_data)

#         # Step 5: Return results
#         return SearchResponse(category=category, confidence=confidence, products=final_products)

#     except Exception as e:
#         logger.error(f"Search error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# @api_router.post("/search", response_model=SearchResponse)
# async def search_products(search_data: SearchRequest, current_user: dict = Depends(get_current_user)):
#     try:
#         category, confidence = classify_query(search_data.query)
#         products = await fetch_products_from_rapidapi(search_data.query)
#         filtered_products = apply_filters(products, search_data.filters)
#         await save_search_history(current_user["id"], search_data)
#         return SearchResponse(category=category, confidence=confidence, products=filtered_products)
#     except Exception as e:
#         logger.error(f"Search error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@api_router.get("/history")
async def get_search_history(current_user: dict = Depends(get_current_user)):
    history = await db.search_history.find({"user_id": current_user["id"]}, {"_id": 0}).sort("timestamp", -1).limit(50).to_list(50)
    for item in history:
        if isinstance(item['timestamp'], str):
            item['timestamp'] = datetime.fromisoformat(item['timestamp'])
    return history

@api_router.post("/complementary", response_model=List[ProductResult])
async def complementary_products(
    product_name: str = Query(..., description="Base product to get complementary items for"),
    budget: Optional[float] = Query(None, description="Max total budget for complementary products"),
    current_user: dict = Depends(get_current_user)
):
    """
    Returns complementary products for a given product within a budget limit.
    Fetches real-time product data from RapidAPI.
    """
    products = await get_complementary_products(product_name, budget)
    return products

# ----------------------------
# Middleware and router
# ----------------------------
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Shutdown event
# ----------------------------
@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()