"""
收益率分布模块
提供不同的收益率分布模型实现
"""
from typing import List
from enum import Enum
import numpy as np
from scipy import stats
from dataclasses import dataclass
from config.investment_config import InvestmentConfig


class RateDistributionModel(Enum):
    """收益率分布模型枚举"""
    NORMAL = "正态分布"
    LOGNORMAL = "对数正态分布"
    T_DISTRIBUTION = "t分布"
    CUSTOM = "自定义分布"


@dataclass
class RateDistribution:
    """收益率分布基类"""
    
    config: InvestmentConfig
    
    def generate_yearly_rates(
        self,
        avg_rate: float,
        years: int,
        volatility: float,
        **kwargs
    ) -> List[float]:
        """生成年度收益率
        
        Args:
            avg_rate: 平均收益率
            years: 年数
            volatility: 波动率
            **kwargs: 其他参数
            
        Returns:
            List[float]: 年度收益率列表
        """
        raise NotImplementedError("子类必须实现此方法")


@dataclass
class NormalDistribution(RateDistribution):
    """正态分布模型"""
    
    def generate_yearly_rates(
        self,
        avg_rate: float,
        years: int,
        volatility: float,
        **kwargs
    ) -> List[float]:
        """使用正态分布生成年度收益率
        
        Args:
            avg_rate: 平均收益率
            years: 年数
            volatility: 波动率
            **kwargs: 其他参数
            
        Returns:
            List[float]: 年度收益率列表
        """
        # 生成正态分布的随机数
        yearly_rates = np.random.normal(
            loc=avg_rate / 100,  # 均值
            scale=volatility / 100,  # 标准差
            size=years  # 生成数量
        )
        
        # 限制最低收益率
        min_allowed_rate = self.config.min_allowed_rate / 100
        yearly_rates = np.maximum(yearly_rates, min_allowed_rate)
        
        return yearly_rates.tolist()


@dataclass
class LogNormalDistribution(RateDistribution):
    """对数正态分布模型"""
    
    def generate_yearly_rates(
        self,
        avg_rate: float,
        years: int,
        volatility: float,
        **kwargs
    ) -> List[float]:
        """使用对数正态分布生成年度收益率
        
        Args:
            avg_rate: 平均收益率
            years: 年数
            volatility: 波动率
            **kwargs: 其他参数
            
        Returns:
            List[float]: 年度收益率列表
        """
        # 对数正态分布参数转换
        avg_rate_decimal = avg_rate / 100
        volatility_decimal = volatility / 100
        
        # 计算对数正态分布的mu和sigma参数
        # 对于对数正态分布，均值 = exp(mu + sigma^2/2)
        # 因此 mu = ln(均值) - sigma^2/2
        sigma = volatility_decimal
        mu = np.log(1 + avg_rate_decimal) - (sigma ** 2) / 2
        
        # 生成对数正态分布的随机数
        yearly_rates = np.random.lognormal(
            mean=mu,
            sigma=sigma,
            size=years
        ) - 1  # 转换为收益率
        
        # 限制最低收益率
        min_allowed_rate = self.config.min_allowed_rate / 100
        yearly_rates = np.maximum(yearly_rates, min_allowed_rate)
        
        return yearly_rates.tolist()


@dataclass
class TDistribution(RateDistribution):
    """t分布模型"""
    
    def generate_yearly_rates(
        self,
        avg_rate: float,
        years: int,
        volatility: float,
        **kwargs
    ) -> List[float]:
        """使用t分布生成年度收益率
        
        Args:
            avg_rate: 平均收益率
            years: 年数
            volatility: 波动率
            **kwargs: 其他参数，包括df（自由度）
            
        Returns:
            List[float]: 年度收益率列表
        """
        # 获取自由度参数，默认为配置中的值
        df = kwargs.get('df', self.config.default_t_distribution_df)
        
        # 生成标准t分布的随机数
        t_values = stats.t.rvs(df=df, size=years)
        
        # 转换为所需的均值和标准差
        avg_rate_decimal = avg_rate / 100
        volatility_decimal = volatility / 100
        
        # 调整t分布的值以匹配所需的均值和标准差
        yearly_rates = avg_rate_decimal + volatility_decimal * t_values
        
        # 限制最低收益率
        min_allowed_rate = self.config.min_allowed_rate / 100
        yearly_rates = np.maximum(yearly_rates, min_allowed_rate)
        
        return yearly_rates.tolist()


@dataclass
class CustomDistribution(RateDistribution):
    """自定义分布模型"""
    
    def generate_yearly_rates(
        self,
        avg_rate: float,
        years: int,
        volatility: float,
        **kwargs
    ) -> List[float]:
        """使用自定义分布生成年度收益率
        
        Args:
            avg_rate: 平均收益率
            years: 年数
            volatility: 波动率
            **kwargs: 其他参数，包括custom_generator（自定义生成器函数）
            
        Returns:
            List[float]: 年度收益率列表
        """
        # 获取自定义生成器函数
        custom_generator = kwargs.get('custom_generator')
        
        if custom_generator is None:
            # 如果没有提供自定义生成器，则使用正态分布作为后备
            return NormalDistribution(self.config).generate_yearly_rates(
                avg_rate=avg_rate,
                years=years,
                volatility=volatility
            )
        
        # 使用自定义生成器生成收益率
        yearly_rates = custom_generator(
            avg_rate=avg_rate / 100,
            years=years,
            volatility=volatility / 100
        )
        
        # 确保结果是列表类型
        if not isinstance(yearly_rates, list):
            yearly_rates = list(yearly_rates)
        
        # 限制最低收益率
        min_allowed_rate = self.config.min_allowed_rate / 100
        yearly_rates = [max(rate, min_allowed_rate) for rate in yearly_rates]
        
        return yearly_rates


def get_distribution_model(
    model_name: str,
    config: InvestmentConfig
) -> RateDistribution:
    """获取收益率分布模型
    
    Args:
        model_name: 模型名称或RateDistributionModel枚举
        config: 投资配置
        
    Returns:
        RateDistribution: 收益率分布模型实例
    """
    # 如果输入是枚举值，转换为字符串
    if isinstance(model_name, RateDistributionModel):
        model_name = model_name.value
    
    # 映射模型名称到分布类
    distribution_models = {
        RateDistributionModel.NORMAL.value: NormalDistribution(config),
        RateDistributionModel.LOGNORMAL.value: LogNormalDistribution(config),
        RateDistributionModel.T_DISTRIBUTION.value: TDistribution(config),
        RateDistributionModel.CUSTOM.value: CustomDistribution(config)
    }
    
    # 返回对应的分布模型
    if model_name not in distribution_models:
        raise ValueError(f"不支持的分布模型: {model_name}")
    
    return distribution_models[model_name]


def generate_yearly_rates(
    avg_rate: float,
    years: int,
    volatility: float,
    distribution_model: str,
    config: InvestmentConfig,
    **kwargs
) -> List[float]:
    """生成年度收益率
    
    Args:
        avg_rate: 平均收益率
        years: 年数
        volatility: 波动率
        distribution_model: 分布模型名称
        config: 投资配置
        **kwargs: 其他参数
        
    Returns:
        List[float]: 年度收益率列表
    """
    # 获取分布模型
    model = get_distribution_model(distribution_model, config)
    
    # 生成收益率
    return model.generate_yearly_rates(
        avg_rate=avg_rate,
        years=years,
        volatility=volatility,
        **kwargs
    )
