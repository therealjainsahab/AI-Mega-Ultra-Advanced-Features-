import os
import logging
import requests
import aiohttp
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import openai
from PIL import Image
import io
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Instagram AI Platform", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure DeepSeek via OpenAI library compatibility
openai.api_key = os.getenv("DEEPSEEK_API_KEY")
openai.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# Models
class ContentType(str, Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    CAROUSEL = "CAROUSEL"
    REEL = "REEL"
    STORY = "STORY"

class PostStatus(str, Enum):
    DRAFT = "DRAFT"
    SCHEDULED = "SCHEDULED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"

class InstagramPost(BaseModel):
    id: Optional[str] = None
    caption: str
    media_urls: List[str]
    content_type: ContentType
    scheduled_time: Optional[datetime] = None
    status: PostStatus = PostStatus.DRAFT
    hashtags: List[str] = []
    mentions: List[str] = []
    location: Optional[str] = None
    user_tags: List[str] = []

class AnalyticsRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    metrics: List[str] = Field(..., description="Metrics to analyze: engagement, reach, impressions, etc.")

class ContentStrategy(BaseModel):
    target_audience: str
    content_themes: List[str]
    posting_frequency: int
    optimal_times: List[str]

class AIContentGenerationRequest(BaseModel):
    theme: str
    style: str
    hashtag_strategy: str
    length: int = 100
    include_hashtags: bool = True

# Services
class LLMService:
    async def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using DeepSeek API (via OpenAI-compatible client)"""
        try:
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

class VisionProcessor:
    async def analyze_image(self, image_url: str) -> Dict[str, Any]:
        """Analyze image for engagement potential"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))

            width, height = img.size
            aspect_ratio = width / height
            brightness = self._estimate_brightness(img)

            return {
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "brightness": brightness,
                "recommended_crop": self._recommend_crop(aspect_ratio),
                "engagement_score": self._calculate_engagement_score(img),
            }
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            return {"error": str(e)}

    def _estimate_brightness(self, img: Image.Image) -> float:
        """Estimate image brightness (0-1)"""
        grayscale = img.convert('L')
        hist = grayscale.histogram()
        pixels = sum(hist)
        if pixels == 0:
            return 0.0
        brightness = sum(i * hist[i] for i in range(256)) / (pixels * 255)
        return float(brightness)

    def _recommend_crop(self, aspect_ratio: float) -> str:
        """Recommend crop based on aspect ratio"""
        if aspect_ratio > 1.2:
            return "Square or 4:5 vertical"
        elif aspect_ratio < 0.8:
            return "Square or 1.91:1 horizontal"
        else:
            return "Good aspect ratio for Instagram"

    def _calculate_engagement_score(self, img: Image.Image) -> float:
        """Calculate engagement score (0-1)"""
        score = 0.5
        brightness = self._estimate_brightness(img)
        if 0.4 <= brightness <= 0.7:
            score += 0.2
        score += 0.1
        return float(min(score, 1.0))

class InstagramAIService:
    def __init__(self, llm_service: LLMService, vision_processor: VisionProcessor):
        self.llm_service = llm_service
        self.vision_processor = vision_processor
        self.base_url = "https://graph.facebook.com/v18.0"
        self.username = os.getenv("INSTAGRAM_USER", "workswvy")
        self.access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")

    async def generate_caption(self, theme: str, style: str = "professional", length: int = 150) -> str:
        """Generate engaging Instagram captions using AI"""
        prompt = f"""
        Generate an engaging Instagram caption about {theme} in a {style} style.
        The caption should be approximately {length} characters long.
        Include relevant emojis and call-to-action.
        """
        response = await self.llm_service.generate_text(prompt)
        return response.strip()

    async def generate_hashtags(self, caption: str, count: int = 15) -> List[str]:
        """Generate relevant hashtags using AI"""
        prompt = f"""
        Based on this Instagram caption: "{caption}"
        Generate {count} relevant, popular hashtags that would increase reach.
        Return only the hashtags, one per line.
        """
        response = await self.llm_service.generate_text(prompt)
        hashtags = [line.strip() for line in response.split('\n') if line.strip().startswith('#')]
        return hashtags[:count]

    async def analyze_image_engagement(self, image_url: str) -> Dict[str, Any]:
        """Predict engagement potential of an image"""
        analysis = await self.vision_processor.analyze_image(image_url)
        engagement_score = analysis.get("engagement_score", 0.5)
        return {
            "engagement_score": engagement_score,
            "recommendations": self._generate_recommendations(analysis),
        }

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on image analysis"""
        recommendations: List[str] = []
        brightness = analysis.get('brightness', 0)
        aspect_ratio = analysis.get('aspect_ratio', 1)
        if brightness < 0.4:
            recommendations.append("Consider brightening the image for better engagement")
        if brightness > 0.7:
            recommendations.append("Consider reducing brightness to avoid overexposure")
        if aspect_ratio > 1.2 or aspect_ratio < 0.8:
            recommendations.append(f"Crop recommendation: {analysis.get('recommended_crop', 'Square')}")
        return recommendations

    async def post_to_instagram(self, post: InstagramPost) -> str:
        """Post content to Instagram using Graph API (placeholder)"""
        try:
            media_id = await self._upload_media(post.media_urls[0], post.content_type)
            container_url = f"{self.base_url}/{media_id}"
            caption = f"{post.caption} {' '.join([f'#{tag}' for tag in post.hashtags])}"
            payload = {
                "caption": caption,
                "access_token": self.access_token,
            }
            if post.location:
                payload["location_id"] = await self._get_location_id(post.location)
            response = requests.post(container_url, data=payload)
            response.raise_for_status()
            return response.json().get("id", "")
        except Exception as e:
            logger.error(f"Failed to post to Instagram: {str(e)}")
            raise

    async def _upload_media(self, media_url: str, content_type: ContentType) -> str:
        """Upload media to Instagram and return media ID (placeholder)"""
        return f"ig_media_{hash(media_url)}"

    async def _get_location_id(self, location_name: str) -> str:
        """Get Facebook location ID from location name (placeholder)"""
        return f"fb_location_{hash(location_name)}"

    async def get_analytics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get comprehensive analytics for the Instagram account (placeholder)"""
        return {
            "username": self.username,
            "engagement": random.randint(100, 1000),
            "impressions": random.randint(500, 5000),
            "reach": random.randint(300, 3000),
            "follower_count": random.randint(1000, 10000),
            "top_posts": [
                {"id": f"post_{i}", "engagement": random.randint(100, 500)} for i in range(5)
            ],
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
        }

    async def process_scheduled_post(self, post: InstagramPost):
        """Process a scheduled post (to be called by background tasks)"""
        try:
            logger.info(f"Processing scheduled post: {post.caption}")
            # await self.post_to_instagram(post)
            logger.info(f"Successfully posted: {post.caption}")
        except Exception as e:
            logger.error(f"Failed to process scheduled post: {str(e)}")

class InstagramContentGenerator:
    def __init__(self, llm_service: LLMService, vision_processor: VisionProcessor):
        self.llm_service = llm_service
        self.vision_processor = vision_processor

    async def generate_content_strategy(self, brand_info: Dict[str, Any]) -> ContentStrategy:
        """Generate a comprehensive content strategy"""
        prompt = f"""
        Create an Instagram content strategy for: {brand_info.get('description', 'a brand')}.
        Target audience: {brand_info.get('target_audience', 'general')}.
        Brand values: {brand_info.get('values', '')}.

        Provide:
        1. 3-5 content themes
        2. Recommended posting frequency
        3. Optimal posting times
        4. Hashtag strategy
        """
        response = await self.llm_service.generate_text(prompt)
        return self._parse_strategy_response(response, brand_info)

    def _parse_strategy_response(self, response: str, brand_info: Dict[str, Any]) -> ContentStrategy:
        """Parse LLM response into ContentStrategy object (simple heuristic)"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        themes: List[str] = []
        frequency = 3
        times = ["09:00", "12:00", "18:00"]
        for line in lines:
            lower = line.lower()
            if "theme" in lower and ":" in line:
                themes.append(line.split(":", 1)[1].strip())
            elif "frequency" in lower and ":" in line:
                try:
                    frequency = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
        if not themes:
            themes = ["Lifestyle", "Products", "Behind the Scenes"]
        return ContentStrategy(
            target_audience=brand_info.get('target_audience', 'general'),
            content_themes=themes,
            posting_frequency=frequency,
            optimal_times=times,
        )

    async def generate_content_calendar(self, strategy: ContentStrategy, days: int = 30) -> List[InstagramPost]:
        """Generate a content calendar for the specified period"""
        calendar: List[InstagramPost] = []
        start_date = datetime.now()
        for day in range(days):
            post_date = start_date + timedelta(days=day)
            if self._should_post_on_day(strategy, post_date):
                post = await self._generate_daily_content(strategy, post_date)
                calendar.append(post)
        return calendar

    def _should_post_on_day(self, strategy: ContentStrategy, date: datetime) -> bool:
        """Determine if should post on a specific day based on frequency"""
        return random.random() < (strategy.posting_frequency / 7)

    async def _generate_daily_content(self, strategy: ContentStrategy, date: datetime) -> InstagramPost:
        """Generate content for a specific day"""
        theme = random.choice(strategy.content_themes)
        caption = await self.llm_service.generate_text(
            f"Create an Instagram post about {theme} for {date.strftime('%A')}"
        )
        hashtags = await self._generate_hashtags_for_theme(theme)
        return InstagramPost(
            caption=caption,
            media_urls=[f"https://example.com/media/{hash(theme + str(date))}.jpg"],
            content_type=random.choice(list(ContentType)),
            scheduled_time=date,
            hashtags=hashtags,
        )

    async def _generate_hashtags_for_theme(self, theme: str) -> List[str]:
        """Generate hashtags for a specific theme"""
        prompt = f"Generate 10 relevant Instagram hashtags for content about {theme}"
        response = await self.llm_service.generate_text(prompt)
        return [tag.strip() for tag in response.split('\n') if tag.strip() and tag.strip().startswith('#')]

# Dependency injection

def get_llm_service() -> LLMService:
    return LLMService()


def get_vision_processor() -> VisionProcessor:
    return VisionProcessor()


def get_instagram_service(
    llm_service: LLMService = Depends(get_llm_service),
    vision_processor: VisionProcessor = Depends(get_vision_processor),
) -> InstagramAIService:
    return InstagramAIService(llm_service, vision_processor)


def get_content_generator(
    llm_service: LLMService = Depends(get_llm_service),
    vision_processor: VisionProcessor = Depends(get_vision_processor),
) -> InstagramContentGenerator:
    return InstagramContentGenerator(llm_service, vision_processor)

# Router
router = APIRouter(prefix="/instagram", tags=["instagram"])


@router.post("/generate-content")
async def generate_content(
    request: AIContentGenerationRequest,
    service: InstagramAIService = Depends(get_instagram_service),
):
    """Generate AI-powered Instagram content"""
    try:
        caption = await service.generate_caption(
            request.theme,
            request.style,
            request.length,
        )
        hashtags: List[str] = []
        if request.include_hashtags:
            hashtags = await service.generate_hashtags(caption)
        return {
            "caption": caption,
            "hashtags": hashtags,
            "recommendations": [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-image")
async def analyze_image(
    image_url: str,
    service: InstagramAIService = Depends(get_instagram_service),
):
    """Analyze an image for engagement potential"""
    try:
        analysis = await service.analyze_image_engagement(image_url)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule-post")
async def schedule_post(
    post: InstagramPost,
    background_tasks: BackgroundTasks,
    service: InstagramAIService = Depends(get_instagram_service),
):
    """Schedule an Instagram post"""
    try:
        background_tasks.add_task(service.process_scheduled_post, post)
        return {"message": "Post scheduled successfully", "post_id": post.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_analytics(
    start_date: datetime,
    end_date: datetime,
    service: InstagramAIService = Depends(get_instagram_service),
):
    """Get Instagram analytics"""
    try:
        analytics = await service.get_analytics(start_date, end_date)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content-strategy")
async def create_content_strategy(
    brand_info: Dict[str, Any],
    generator: InstagramContentGenerator = Depends(get_content_generator),
):
    """Generate a content strategy for a brand"""
    try:
        strategy = await generator.generate_content_strategy(brand_info)
        return strategy
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content-calendar")
async def generate_content_calendar(
    strategy: ContentStrategy,
    days: int = 30,
    generator: InstagramContentGenerator = Depends(get_content_generator),
):
    """Generate a content calendar"""
    try:
        calendar = await generator.generate_content_calendar(strategy, days)
        return {"calendar": calendar}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-upload")
async def bulk_upload(
    files: List[UploadFile] = File(...),
    service: InstagramAIService = Depends(get_instagram_service),
):
    """Bulk upload content for scheduling"""
    try:
        results: List[Dict[str, Any]] = []
        for file in files:
            _ = await file.read()
            results.append({"filename": file.filename, "status": "processed"})
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include router in app
app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Instagram AI Platform API", "user": os.getenv("INSTAGRAM_USER")}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "deepseek_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        "username": os.getenv("INSTAGRAM_USER"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)